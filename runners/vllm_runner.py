import os
import sqlparse
import time
import torch
import pandas as pd
from typing import List
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

from utils.gen_prompt import generate_prompt
from utils.questions import prepare_questions_df
from utils.creds import db_creds_all
from utils.reporting import upload_results
from eval.eval import compare_query_results


def run_vllm_eval(args):
    """Run evaluation using VLLM with batching"""
    questions_file_list = args.questions_file
    prompt_file_list = args.prompt_file
    output_file_list = args.output_file
    num_questions = args.num_questions
    public_data = not args.use_private_data
    model_name = args.model
    num_beams = args.num_beams
    k_shot = args.k_shot
    db_type = args.db_type
    cot_table_alias = args.cot_table_alias

    # VLLM-specific LoRA handling
    enable_lora = True if args.adapter else False
    lora_request = LoRARequest("sql_adapter", 1, args.adapter) if args.adapter else None

    # Initialize VLLM model and tokenizer
    print(f"Preparing {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # VLLM-specific model initialization
    if not args.quantized:
        llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            enable_lora=enable_lora,
            max_model_len=4096,
            max_lora_rank=64,
        )
    else:
        llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            quantization="AWQ",
            enable_lora=enable_lora,
            max_model_len=4096,
            max_lora_rank=64,
        )

    sampling_params = SamplingParams(
        n=1,
        best_of=num_beams,
        use_beam_search=num_beams != 1,
        stop_token_ids=[tokenizer.eos_token_id],
        max_tokens=1000,
        temperature=0,
    )

    for questions_file, prompt_file, output_file in zip(
        questions_file_list, prompt_file_list, output_file_list
    ):
        print(f"Using prompt file {prompt_file}")
        print("Preparing questions...")
        print(
            f"Using {'all' if num_questions is None else num_questions} question(s) from {questions_file}"
        )
        df = prepare_questions_df(
            questions_file, db_type, num_questions, k_shot, cot_table_alias
        )

        # Create prompts with all parameters
        df["prompt"] = df.apply(
            lambda row: generate_prompt(
                prompt_file,
                row["question"],
                row["db_name"],
                row["db_type"],
                row["instructions"],
                row["k_shot_prompt"],
                row["glossary"],
                row["table_metadata_string"],
                row["prev_invalid_sql"],
                row["prev_error_msg"],
                row.get("question_0", ""),
                row.get("query_0", ""),
                row.get("question_1", ""),
                row.get("query_1", ""),
                row.get("cot_instructions", ""),
                row.get("cot_pregen", False),
                public_data,
                args.num_columns if hasattr(args, "num_columns") else 40,
                args.shuffle_metadata,
                row.get("table_aliases", ""),
            ),
            axis=1,
        )
        print(f"Prepared {len(df)} question(s) from {questions_file}")

        def chunk_dataframe(df, chunk_size):
            """Returns successive chunk_size chunks from df as a list of dfs"""
            df_chunks = []
            for i in range(0, len(df), chunk_size):
                df_i = df.iloc[i : min(i + chunk_size, len(df))]
                print(
                    f"Chunk {i//chunk_size+1}/{len(df)//chunk_size+1} with {len(df_i)} questions"
                )
                df_chunks.append(df_i)
            return df_chunks

        # VLLM-specific batch processing
        df_chunks = chunk_dataframe(df, args.batch_size)
        total_tried = 0
        total_correct = 0
        output_rows = []

        print("Generating completions")
        for batch in (pbar := tqdm(df_chunks, total=len(df))):
            prompts = batch["prompt"].tolist()
            print(f"Generating completions for {len(prompts)} prompts")

            # VLLM-specific token handling
            prompt_tokens = []
            prompt_token_sizes = []
            for prompt in prompts:
                token_ids = tokenizer.encode(prompt, add_special_tokens=False)
                if token_ids[0] != tokenizer.bos_token_id:
                    token_ids = [tokenizer.bos_token_id] + token_ids
                prompt_tokens.append(token_ids)
                prompt_token_sizes.append(len(token_ids))
            print(
                f"Average prompt size: {sum(prompt_token_sizes)/len(prompt_token_sizes):.0f}"
            )

            start_time = time.time()
            outputs = llm.generate(
                sampling_params=sampling_params,
                prompt_token_ids=prompt_tokens,
                use_tqdm=False,
                lora_request=lora_request,
            )
            print(
                f"Generated {len(outputs)} completions in {time.time() - start_time:.2f} seconds"
            )
            time_taken = time.time() - start_time

            for row, output in zip(batch.to_dict("records"), outputs):
                generated_query = (
                    output.outputs[0].text.split(";")[0].split("```")[0].strip() + ";"
                )
                normalized_query = sqlparse.format(
                    generated_query, keyword_case="upper", strip_whitespace=True
                )
                row["generated_query"] = normalized_query
                row["tokens_used"] = len(output.outputs[0].token_ids)
                row["latency_seconds"] = time_taken / len(batch)

                # Verify results
                golden_query = row["query"]
                db_name = row["db_name"]
                db_type = row["db_type"]
                question = row["question"]
                query_category = row["query_category"]
                table_metadata_string = row["table_metadata_string"]

                try:
                    exact_match, correct = compare_query_results(
                        query_gold=golden_query,
                        query_gen=generated_query,
                        db_name=db_name,
                        db_type=db_type,
                        db_creds=db_creds_all[db_type],
                        question=question,
                        query_category=query_category,
                        table_metadata_string=table_metadata_string,
                        decimal_points=(
                            args.decimal_points
                            if hasattr(args, "decimal_points")
                            else 2
                        ),
                    )
                    row["exact_match"] = int(exact_match)
                    row["correct"] = int(correct)
                    row["is_correct"] = int(correct)  # For base runner compatibility
                    row["error_msg"] = ""
                    if correct:
                        total_correct += 1
                except Exception as e:
                    row["error_db_exec"] = 1
                    row["error_msg"] = f"QUERY EXECUTION ERROR: {e}"

                total_tried += 1
                output_rows.append(row)

            pbar.update(len(batch))
            pbar.set_description(
                f"Correct so far: {total_correct}/{total_tried} ({100*total_correct/total_tried:.2f}%)"
            )

        # Process results
        df = pd.DataFrame(output_rows)
        if "prompt" in df.columns:
            del df["prompt"]

        # Get stats by query category
        agg_stats = df.groupby("query_category")[["exact_match", "correct"]].mean()
        print(agg_stats)
        df = df.sort_values(by=["db_name", "query_category", "question"])
        print(f"Average tokens generated: {df['tokens_used'].mean():.1f}")

        # Create output directory if needed
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        df.to_csv(output_file, index=False, float_format="%.2f")
        print(f"Saved results to {output_file}")

        # Print summary stats
        print(f"Total questions: {total_tried}")
        print(f"Total correct: {total_correct}")
        print(f"Accuracy: {total_correct/total_tried:.3f}")

        # Upload results if URL provided
        try:
            if hasattr(args, "upload_url") and args.upload_url:
                with open(prompt_file, "r") as f:
                    prompt = f.read()
                upload_results(
                    results=df.to_dict("records"),
                    url=args.upload_url,
                    runner_type="vllm_runner",
                    prompt=prompt,
                    args=args,
                )
        except Exception as e:
            print(f"Error uploading results: {e}")
