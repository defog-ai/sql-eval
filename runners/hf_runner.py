import os
from typing import Optional
import torch
import gc
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)
from tqdm import tqdm
from psycopg2.extensions import QueryCanceledError

from utils.gen_prompt import generate_prompt
from utils.questions import prepare_questions_df
from utils.creds import db_creds_all
from utils.reporting import upload_results
from eval.eval import compare_query_results

device_map = "mps" if torch.backends.mps.is_available() else "auto"

def get_tokenizer_model(model_name: Optional[str], adapter_path: Optional[str]):
    """
    Load a HuggingFace tokenizer and model.
    You may supply either a normal huggingface model name, or a peft adapter path.
    """
    if adapter_path is not None:
        from peft import PeftModel, PeftConfig

        print(f"Loading adapter model {adapter_path}")
        config = PeftConfig.from_pretrained(adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            use_cache=True,
            device_map=device_map,
        )
        print(f"Loading adapter {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print(f"Merged adapter {adapter_path}")
    else:
        print(f"Loading model {model_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except:
            tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Meta-Llama-3-8B-Instruct"
            )

        tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map=device_map,
        )
    return tokenizer, model


def extract_hf_sql(text: str, has_sql_tag: bool) -> str:
    """HuggingFace-specific SQL extraction"""
    if not has_sql_tag:
        return text.split("```")[0].split(";")[0].strip() + ";"
    else:
        return text.split("[/SQL]")[0].split(";")[0].strip() + ";"


def run_hf_eval(args):
    """Run evaluation using HuggingFace models"""
    questions_file_list = args.questions_file
    prompt_file_list = args.prompt_file
    output_file_list = args.output_file
    num_questions = args.num_questions
    public_data = not args.use_private_data
    model_name = args.model
    adapter_path = args.adapter
    k_shot = args.k_shot
    db_type = args.db_type
    num_beams = args.num_beams
    cot_table_alias = args.cot_table_alias

    if model_name is None and adapter_path is None:
        raise ValueError(
            "You must supply either a model name or an adapter path to run an evaluation."
        )

    print(f"Questions prepared\nNow loading model...")
    # initialize tokenizer and model
    tokenizer, model = get_tokenizer_model(model_name, adapter_path)

    if "8b" in model_name.lower():
        # do this since it doesn't seem to have been done by default
        tokenizer.padding_side = "left"

    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.tie_weights()

    print("model loaded\nnow generating and evaluating predictions...")

    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, batch_size=args.batch_size
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
                args.num_columns if hasattr(args, 'num_columns') else 40,
                args.shuffle_metadata,
                row.get("table_aliases", ""),
            ),
            axis=1,
        )

        total_tried = 0
        total_correct = 0
        output_rows = []

        def chunk_dataframe(df, chunk_size):
            """Yield successive chunk_size chunks from df."""
            for i in range(0, len(df), chunk_size):
                yield df[i : min(i + chunk_size, len(df))]

        df_chunks = list(chunk_dataframe(df, args.batch_size))

        with tqdm(total=len(df)) as pbar:
            for batch in df_chunks:
                prompts = batch["prompt"].tolist()
                generated_queries = pipe(
                    prompts,
                    max_new_tokens=600,
                    do_sample=False,
                    num_beams=num_beams,
                    num_return_sequences=1,
                    return_full_text=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                    temperature=None,
                    top_p=None,
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                for row, result in zip(batch.to_dict("records"), generated_queries):
                    total_tried += 1
                    has_sql_tag = "[SQL]" in row["prompt"]
                    generated_query = extract_hf_sql(
                        result[0]["generated_text"], has_sql_tag
                    )

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

                    row["generated_query"] = generated_query
                    row["latency_seconds"] = None
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
                            decimal_points=args.decimal_points if hasattr(args, 'decimal_points') else 2,
                        )
                        row["exact_match"] = int(exact_match)
                        row["correct"] = int(correct)
                        row["is_correct"] = int(correct)  # For base runner compatibility
                        row["error_msg"] = ""
                        if correct:
                            total_correct += 1
                    except QueryCanceledError as e:
                        row["timeout"] = 1
                        row["error_msg"] = f"QUERY EXECUTION TIMEOUT: {e}"
                    except Exception as e:
                        row["error_db_exec"] = 1
                        row["error_msg"] = f"QUERY EXECUTION ERROR: {e}"

                    output_rows.append(row)
                    pbar.update(1)
                    pbar.set_description(
                        f"Correct so far: {total_correct}/{total_tried} ({100*total_correct/total_tried:.2f}%)"
                    )

        # Convert to DataFrame and save results
        output_df = pd.DataFrame(output_rows)
        output_df = output_df.sort_values(by=["db_name", "query_category", "question"])
        if "prompt" in output_df.columns:
            del output_df["prompt"]

        # Get stats by query category
        agg_stats = (
            output_df.groupby("query_category")
            .agg(
                num_rows=("db_name", "count"),
                mean_correct=("correct", "mean"),
                mean_error_db_exec=("error_db_exec", "mean"),
            )
            .reset_index()
        )
        print(agg_stats)

        # Create output directory if needed
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        output_df.to_csv(output_file, index=False, float_format="%.2f")

        # Print summary stats
        print(f"Total questions: {total_tried}")
        print(f"Total correct: {total_correct}")
        print(f"Accuracy: {total_correct/total_tried:.3f}")

        # Upload results if URL provided
        try:
            if hasattr(args, 'upload_url') and args.upload_url:
                with open(prompt_file, "r") as f:
                    prompt = f.read()
                upload_results(
                    results=output_df.to_dict("records"),
                    url=args.upload_url,
                    runner_type="hf_runner",
                    prompt=prompt,
                    args=args,
                )
        except Exception as e:
            print(f"Error uploading results: {e}")