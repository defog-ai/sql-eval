import json
import os
from typing import List
import sqlparse
from vllm import LLM, SamplingParams
from eval.eval import compare_query_results
import pandas as pd
from utils.gen_prompt import generate_prompt
from utils.questions import prepare_questions_df
from utils.creds import db_creds_all
import time
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
from utils.reporting import upload_results


def run_vllm_eval(args):
    # get params from args
    questions_file_list = args.questions_file
    prompt_file_list = args.prompt_file
    num_questions = args.num_questions
    public_data = not args.use_private_data
    model_name = args.model
    output_file_list = args.output_file
    num_beams = args.num_beams
    k_shot = args.k_shot
    db_type = args.db_type
    cot_table_alias = args.cot_table_alias

    # initialize model only once as it takes a while
    print(f"Preparing {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if not args.quantized:
        llm = LLM(model=model_name, tensor_parallel_size=1)
    else:
        llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            quantization="AWQ",
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
        # get questions
        print("Preparing questions...")
        print(
            f"Using {'all' if num_questions is None else num_questions} question(s) from {questions_file}"
        )
        df = prepare_questions_df(
            questions_file, db_type, num_questions, k_shot, cot_table_alias
        )
        # create a prompt for each question
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
                row["question_0"],
                row["query_0"],
                row["question_1"],
                row["query_1"],
                row["cot_instructions"],
                row["cot_pregen"],
                public_data,
                args.num_columns,
                args.shuffle_metadata,
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

        df_chunks = chunk_dataframe(df, args.batch_size)

        total_tried = 0
        total_correct = 0
        output_rows = []

        print(f"Generating completions")

        for batch in (pbar := tqdm(df_chunks, total=len(df))):
            prompts = batch["prompt"].tolist()
            print(f"Generating completions for {len(prompts)} prompts")
            prompt_tokens = []
            prompt_token_sizes = []
            for prompt in prompts:
                token_ids = tokenizer.encode(prompt, add_special_tokens=False)
                # add bos token if not already present in prompt
                if token_ids[0] != tokenizer.bos_token_id:
                    token_ids = [tokenizer.bos_token_id] + token_ids
                prompt_tokens.append(token_ids)
                prompt_token_sizes.append(len(token_ids))
            print(
                f"Average prompt size: {sum(prompt_token_sizes)/len(prompt_token_sizes):.0f}"
            )
            start_time = time.time()
            # outputs = llm.generate(prompts, sampling_params) # if you prefer to use prompts instead of token_ids
            outputs = llm.generate(
                sampling_params=sampling_params,
                prompt_token_ids=prompt_tokens,
                use_tqdm=False,
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

                golden_query = row["query"]
                db_name = row["db_name"]
                db_type = row["db_type"]
                question = row["question"]
                query_category = row["query_category"]
                table_metadata_string = row["table_metadata_string"]
                exact_match = correct = 0
                db_creds = db_creds_all[db_type]
                try:
                    exact_match, correct = compare_query_results(
                        query_gold=golden_query,
                        query_gen=generated_query,
                        db_name=db_name,
                        db_type=db_type,
                        db_creds=db_creds,
                        question=question,
                        query_category=query_category,
                        table_metadata_string=table_metadata_string,
                        decimal_points=args.decimal_points,
                    )
                    row["exact_match"] = int(exact_match)
                    row["correct"] = int(correct)
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
                f"Correct so far: {total_correct}/{(total_tried)} ({100*total_correct/(total_tried):.2f}%)"
            )
        df = pd.DataFrame(output_rows)
        del df["prompt"]
        print(df.groupby("query_category")[["exact_match", "correct"]].mean())
        df = df.sort_values(by=["db_name", "query_category", "question"])
        print(f"Average tokens generated: {df['tokens_used'].mean():.1f}")
        # get directory of output_file and create if not exist
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        df.to_csv(output_file, index=False, float_format="%.2f")
        print(f"Saved results to {output_file}")

        results = df.to_dict("records")
        # upload results
        with open(prompt_file, "r") as f:
            prompt = f.read()
        if args.upload_url is not None:
            upload_results(
                results=results,
                url=args.upload_url,
                runner_type="vllm_runner",
                prompt=prompt,
                args=args,
            )
