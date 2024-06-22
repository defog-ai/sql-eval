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


def mk_vllm_json(prompt, num_beams, logprobs=False, sql_lora_path=None):
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
):
    start_time = time()
    if api_type == "tgi":
        json_data = mk_tgi_json(row["prompt"], num_beams)
    elif api_type == "vllm":
        json_data = mk_vllm_json(row["prompt"], num_beams, logprobs, sql_lora_path)
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
            timeout=15,
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

    # remove extra spaces around brackets especially for MySQL
    generated_query = generated_query.replace(" ( ", "(").replace(" )", ")")

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
    # get params from args
    questions_file_list = args.questions_file
    prompt_file_list = args.prompt_file
    num_questions = args.num_questions
    public_data = not args.use_private_data
    api_url = args.api_url
    api_type = args.api_type
    output_file_list = args.output_file
    k_shot = args.k_shot
    num_beams = args.num_beams
    max_workers = args.parallel_threads
    db_type = args.db_type
    decimal_points = args.decimal_points
    logprobs = args.logprobs
    cot_table_alias = args.cot_table_alias
    sql_lora_path = args.adapter if args.adapter else None
    run_name = args.run_name if args.run_name else None
    if sql_lora_path:
        print("Using LoRA adapter at:", sql_lora_path)
    if logprobs:
        # check that the eval-visualizer/public directory exists
        if not os.path.exists("./eval-visualizer"):
            # thorow error
            raise Exception(
                "The eval-visualizer directory does not exist. Please clone it with `git clone https://github.com/defog-ai/eval-visualizer/` before running sql-eval with the --logprobs flag."
            )

        if not os.path.exists("./eval-visualizer/public"):
            os.makedirs("./eval-visualizer/public")

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

        total_tried = 0
        total_correct = 0
        output_rows = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for row in df.to_dict("records"):
                futures.append(
                    executor.submit(
                        process_row,
                        row,
                        api_url,
                        api_type,
                        num_beams,
                        decimal_points,
                        logprobs,
                        sql_lora_path,
                    )
                )

            with tqdm(as_completed(futures), total=len(futures)) as pbar:
                for f in pbar:
                    row = f.result()
                    output_rows.append(row)
                    if row["correct"]:
                        total_correct += 1
                    total_tried += 1
                    pbar.update(1)
                    pbar.set_description(
                        f"Correct so far: {total_correct}/{total_tried} ({100*total_correct/total_tried:.2f}%)"
                    )

        output_df = pd.DataFrame(output_rows)

        print(output_df.groupby("query_category")[["correct", "error_db_exec"]].mean())
        output_df = output_df.sort_values(by=["db_name", "query_category", "question"])
        # get directory of output_file and create if not exist
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if logprobs:
            print(
                f"Writing logprobs to JSON file at eval-visualizer/public/{output_file.split('/')[-1].replace('.csv', '.json')}"
            )
            results = output_df.to_dict("records")
            with open(
                f"./eval-visualizer/public/{output_file.split('/')[-1].replace('.csv', '.json')}",
                "w",
            ) as f:
                json.dump(results, f)

        del output_df["prompt"]
        try:
            output_df.to_csv(output_file, index=False, float_format="%.2f")
        except:
            output_df.to_pickle(output_file)

        # upload results
        # with open(prompt_file, "r") as f:
        #     prompt = f.read()

        if run_name is None:
            run_name = output_file.split("/")[-1].replace(".csv", "")
            print(
                "Run name not provided. Using a output filename for run name:", run_name
            )

        if args.upload_url is not None:
            upload_results(
                results=results,
                url=args.upload_url,
                runner_type="api_runner",
                args=args,
                run_name=run_name,
            )
