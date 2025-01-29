import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from eval.eval import compare_query_results
import pandas as pd
from utils.gen_prompt import generate_prompt
from utils.questions import prepare_questions_df
from utils.creds import db_creds_all
from tqdm import tqdm
from time import time
import requests
from utils.reporting import upload_results
import sqlparse
import re


def clean_generated_query(query: str):
    """
    Clean up the generated query by
    - formatting the query using sqlparse
    - fixing common problems in LLM-powered query generation with post-processing heuristics

    KNOWN ISSUES: the division fix will only work with Postgres/Redshift/Snowflake/Databricks. It might not work with other databases.
    """

    query = sqlparse.format(query, reindent_aligned=True)

    # if the string `< =` is present, replace it with `<=`. Similarly for `> =` and `>=`
    query = query.replace("< =", "<=").replace("> =", ">=")

    # if the string ` / NULLIF (` is present, replace it with `/ NULLIF ( 1.0 * `.
    # This is a fix for ensuring that the denominator is always a float in division operations.
    query = query.replace("/ NULLIF (", "/ NULLIF (1.0 * ")

    # remove extra spaces around brackets especially for MySQL
    query = re.sub(r"\s*\(\s*", "(", query)  # Remove spaces before and after '('
    query = re.sub(r"\s*\)", ")", query)  # Remove spaces before ')'

    return query


def mk_vllm_json(
    prompt, num_beams, logprobs=False, sql_lora_path=None, sql_lora_name=None
):
    payload = {
        "prompt": prompt,
        "n": 1,
        "use_beam_search": num_beams > 1,
        "best_of": num_beams,
        "temperature": 0,
        "stop": [";", "```"],
        "max_tokens": 4000,
        "seed": 42,
        "sql_lora_path": sql_lora_path,
        "sql_lora_name": sql_lora_name,
    }
    if logprobs:
        payload["logprobs"] = 2
    return payload


def mk_tgi_json(prompt, num_beams):
    # see swagger docs for /generate for the full list of parameters:
    # https://huggingface.github.io/text-generation-inference/#/Text%20Generation%20Inference/generate
    return {
        "inputs": prompt,
        "parameters": {
            "best_of": num_beams,
            "do_sample": num_beams > 1,
            "return_full_text": False,
            "max_new_tokens": 1024,
        },
    }


def process_row(
    row,
    api_url: str,
    api_type: str,
    num_beams: int,
    decimal_points: int,
    logprobs: bool = False,
    sql_lora_path: Optional[str] = None,
    sql_lora_name: Optional[str] = None,
):
    start_time = time()
    if api_type == "tgi":
        json_data = mk_tgi_json(row["prompt"], num_beams)
    elif api_type == "vllm":
        json_data = mk_vllm_json(
            row["prompt"], num_beams, logprobs, sql_lora_path, sql_lora_name
        )
    else:
        # add any custom JSON data here, e.g. for a custom API
        json_data = {
            "prompt": row["prompt"],
            "n": 1,
            "use_beam_search": num_beams > 1,
            "best_of": num_beams,
            "temperature": 0,
            "stop": [";", "```"],
            "max_tokens": 4000,
        }
    try:
        r = requests.post(
            api_url,
            json=json_data,
            timeout=200,
        )
    except:
        row["generated_query"] = ""
        row["exact_match"] = 0
        row["correct"] = 0
        row["error_db_exec"] = 1
        row["error_msg"] = "API TIMEOUT"
        row["tokens_used"] = None
        if logprobs:
            row["logprobs"] = []

        return row
    end_time = time()
    logprobs = []
    if api_type == "tgi":
        # we do not return the original prompt in tgi
        try:
            generated_query = r.json()["generated_text"]
        except KeyError:
            print(r.json())
            generated_query = ""
    elif "[SQL]" not in row["prompt"]:
        generated_query = (
            r.json()["text"][0]
            .split("```sql")[-1]
            .split("```")[0]
            .split(";")[0]
            .strip()
            + ";"
        )
    else:
        generated_query = r.json()["text"][0]
        if "[SQL]" in generated_query:
            generated_query = generated_query.split("[SQL]", 1)[1].strip()
        else:
            generated_query = generated_query.strip()

    # clean up the generated query
    generated_query = clean_generated_query(generated_query)

    if "logprobs" in r.json():
        logprobs = r.json()["logprobs"]

    row["generated_query"] = generated_query
    logprobs_display = []
    for item in logprobs:
        probs = list(item.values())
        probs_to_append = {}
        for prob in probs:
            rank = prob["rank"]
            logprob = prob["logprob"]
            token = prob["decoded_token"]
            probs_to_append.update(
                {
                    f"rank_{rank}_token": token,
                    f"rank_{rank}_logprob": logprob,
                    f"rank_{rank}_prob": 10**logprob,
                }
            )

        probs_to_append["prob_diff"] = (
            probs_to_append["rank_1_prob"] - probs_to_append["rank_2_prob"]
        )
        logprobs_display.append(probs_to_append)
    row["logprobs"] = logprobs_display
    row["latency_seconds"] = end_time - start_time
    row["tokens_used"] = None
    golden_query = row["query"]
    db_name = row["db_name"]
    db_type = row["db_type"]
    question = row["question"]
    query_category = row["query_category"]
    table_metadata_string = row["table_metadata_string"]
    exact_match = correct = 0

    try:
        exact_match, correct = compare_query_results(
            query_gold=golden_query,
            query_gen=generated_query,
            db_name=db_name,
            db_type=db_type,
            db_creds=db_creds_all.get(row["db_type"], {}),
            question=question,
            query_category=query_category,
            table_metadata_string=table_metadata_string,
            decimal_points=decimal_points,
        )
        row["exact_match"] = int(exact_match)
        row["correct"] = int(correct)
        row["error_msg"] = ""
    except Exception as e:
        row["error_db_exec"] = 1
        row["error_msg"] = f"QUERY EXECUTION ERROR: {e}"

    return row


def run_api_eval(args):
    """Run evaluation using API"""
    questions_file_list = args.questions_file
    prompt_file_list = args.prompt_file
    output_file_list = args.output_file
    num_questions = args.num_questions
    public_data = not args.use_private_data
    k_shot = args.k_shot
    cot_table_alias = args.cot_table_alias
    db_type = args.db_type
    logprobs = args.logprobs
    run_name = getattr(args, "run_name", None)
    sql_lora_path = getattr(args, "adapter", None)

    if sql_lora_path:
        print("Using LoRA adapter at:", sql_lora_path)

    # Logprobs visualization directory handling
    if logprobs:
        if not os.path.exists("./eval-visualizer"):
            raise Exception(
                "The eval-visualizer directory does not exist. Please clone it with "
                "`git clone https://github.com/defog-ai/eval-visualizer/` before running "
                "sql-eval with the --logprobs flag."
            )
        if not os.path.exists("./eval-visualizer/public"):
            os.makedirs("./eval-visualizer/public")

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

        output_rows, total_correct, total_tried = run_eval_in_threadpool(
            df, args.api_url, process_row, args
        )

        output_df = pd.DataFrame(output_rows)
        output_df = output_df.sort_values(by=["db_name", "query_category", "question"])

        # Handle logprobs visualization
        if logprobs:
            results = output_df.to_dict("records")
            print(
                f"Writing logprobs to JSON file at eval-visualizer/public/{output_file.split('/')[-1].replace('.csv', '.json')}"
            )
            with open(
                f"./eval-visualizer/public/{output_file.split('/')[-1].replace('.csv', '.json')}",
                "w",
            ) as f:
                json.dump(results, f)

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

        # Clean up and save results
        if "prompt" in output_df.columns:
            del output_df["prompt"]

        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        try:
            output_df.to_csv(output_file, index=False, float_format="%.2f")
        except:
            output_df.to_pickle(output_file)

        # Handle run naming and result upload
        if run_name is None:
            run_name = output_file.split("/")[-1].replace(".csv", "")
            print(
                "Run name not provided. Using output filename for run name:", run_name
            )

        print(f"Total questions: {total_tried}")
        print(f"Total correct: {total_correct}")
        print(f"Accuracy: {total_correct/total_tried:.3f}")

        try:
            if hasattr(args, "upload_url") and args.upload_url:
                upload_results(
                    results=output_df.to_dict("records"),
                    url=args.upload_url,
                    runner_type="api_runner",
                    args=args,
                    run_name=run_name,
                )
        except Exception as e:
            print(f"Error uploading results: {e}")
