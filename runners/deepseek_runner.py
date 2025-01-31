import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict

from eval.eval import compare_query_results
import pandas as pd
from utils.gen_prompt import generate_prompt
from utils.questions import prepare_questions_df
from utils.creds import db_creds_all
from tqdm import tqdm
from time import time
from openai import OpenAI
from utils.reporting import upload_results


client = OpenAI(
    base_url="https://api.deepseek.com", api_key=os.environ.get("DEEPSEEK_API_KEY")
)


def process_row(row: Dict, model: str):
    start_time = time()
    messages = row["prompt"]
    if model != "deepseek-reasoner":
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=800,
            temperature=0.0,
        )
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=800,
        )
    content = response.choices[0].message.content
    generated_query = content.replace("```sql", "").replace("```", "").strip()
    end_time = time()

    row["generated_query"] = generated_query
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
            db_creds=db_creds_all[row["db_type"]],
            question=question,
            query_category=query_category,
            table_metadata_string=table_metadata_string,
        )
        row["exact_match"] = int(exact_match)
        row["correct"] = int(correct)
        row["error_msg"] = ""
    except Exception as e:
        row["error_db_exec"] = 1
        row["error_msg"] = f"QUERY EXECUTION ERROR: {e}"

    return row


def run_deepseek_eval(args):
    # get params from args
    questions_file_list = args.questions_file
    prompt_file_list = args.prompt_file
    num_questions = args.num_questions
    public_data = not args.use_private_data
    output_file_list = args.output_file
    k_shot = args.k_shot
    max_workers = args.parallel_threads
    db_type = args.db_type
    decimal_points = args.decimal_points
    model = args.model
    cot_table_alias = args.cot_table_alias

    for questions_file, prompt_file, output_file in zip(
        questions_file_list, prompt_file_list, output_file_list
    ):
        if not prompt_file.endswith(".json"):
            raise ValueError(f"Prompt file must be a JSON file. Got {prompt_file}")
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
        # note that the prompt for together ai uses the openai chat API
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
                row["table_aliases"],
            ),
            axis=1,
        )

        total_tried = 0
        total_correct = 0
        output_rows = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for row in df.to_dict("records"):
                futures.append(executor.submit(process_row, row, model))

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
        del output_df["prompt"]
        print(output_df.groupby("query_category")[["correct", "error_db_exec"]].mean())
        output_df = output_df.sort_values(by=["db_name", "query_category", "question"])
        # get directory of output_file and create if not exist
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        try:
            output_df.to_csv(output_file, index=False, float_format="%.2f")
        except:
            output_df.to_pickle(output_file)

        results = output_df.to_dict("records")
        # upload results
        with open(prompt_file, "r") as f:
            prompt = f.read()
        if args.upload_url is not None:
            upload_results(
                results=results,
                url=args.upload_url,
                runner_type="api_runner",
                prompt=prompt,
                args=args,
            )
