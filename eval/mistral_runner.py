from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from time import time
from typing import Optional

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import pandas as pd
from tqdm import tqdm

from eval.eval import compare_query_results
from utils.creds import db_creds_all
from utils.pruning import prune_metadata_str
from utils.questions import prepare_questions_df
from utils.reporting import upload_results

api_key = os.environ.get("MISTRAL_API_KEY")

client = MistralClient(api_key=api_key)


def generate_prompt(
    prompt_file,
    question,
    db_name,
    instructions="",
    k_shot_prompt="",
    glossary="",
    table_metadata_string="",
    prev_invalid_sql="",
    prev_error_msg="",
    public_data=True,
    columns_to_keep=20,
    shuffle=True,
):
    with open(prompt_file, "r") as f:
        prompt = f.read()

    # Check that System and User prompts are in the prompt file
    if "System:" not in prompt or "User:" not in prompt:
        raise ValueError("Invalid prompt file. Please use prompt_mistral.md")
    sys_prompt = prompt.split("System:")[1].split("User:")[0].strip()
    user_prompt = prompt.split("User:")[1].strip()

    question_instructions = question + " " + instructions

    if table_metadata_string == "":
        pruned_metadata_str = prune_metadata_str(
            question_instructions, db_name, public_data, columns_to_keep, shuffle
        )
    else:
        pruned_metadata_str = table_metadata_string

    user_prompt = user_prompt.format(
        user_question=question,
        instructions=instructions,
        table_metadata_string=pruned_metadata_str,
        k_shot_prompt=k_shot_prompt,
        glossary=glossary,
        prev_invalid_sql=prev_invalid_sql,
        prev_error_msg=prev_error_msg,
    )
    messages = [
        ChatMessage(
            role="system",
            content=sys_prompt,
        ),
        ChatMessage(
            role="user",
            content=user_prompt,
        ),
    ]
    return messages


def process_row(row, model, args):
    start_time = time()
    chat_response = client.chat(
        model=model,
        messages=row["prompt"],
        temperature=0,
        max_tokens=600,
    )
    end_time = time()
    generated_query = chat_response.choices[0].message.content

    try:
        # replace all backslashes with empty string
        generated_query = generated_query.replace("\\", "")

        generated_query = generated_query.split(";")[0].split("```sql")[-1].strip()
        generated_query = [i for i in generated_query.split("```") if i.strip() != ""][
            0
        ] + ";"
    except Exception as e:
        print(e)
        generated_query = chat_response.choices[0].message.content
    row["generated_query"] = generated_query
    row["latency_seconds"] = end_time - start_time
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
            decimal_points=args.decimal_points,
        )
        row["exact_match"] = int(exact_match)
        row["correct"] = int(correct)
        row["error_msg"] = ""
    except Exception as e:
        row["error_db_exec"] = 1
        row["error_msg"] = f"QUERY EXECUTION ERROR: {e}"

    return row


def run_mistral_eval(args):
    # get params from args
    questions_file_list = args.questions_file
    prompt_file_list = args.prompt_file
    num_questions = args.num_questions
    public_data = not args.use_private_data
    model = args.model
    output_file_list = args.output_file
    k_shot = args.k_shot
    max_workers = args.parallel_threads
    db_type = args.db_type

    for questions_file, prompt_file, output_file in zip(
        questions_file_list, prompt_file_list, output_file_list
    ):
        print(f"Using prompt file {prompt_file}")
        # get questions
        print("Preparing questions...")
        print(
            f"Using {'all' if num_questions is None else num_questions} question(s) from {questions_file}"
        )
        df = prepare_questions_df(questions_file, db_type, num_questions, k_shot)
        # create a prompt for each question
        df["prompt"] = df[
            [
                "question",
                "db_name",
                "db_type",
                "instructions",
                "k_shot_prompt",
                "glossary",
                "table_metadata_string",
                "prev_invalid_sql",
                "prev_error_msg",
                "question_0",
                "query_0",
                "question_1",
                "query_1",
            ]
        ].apply(
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
                futures.append(executor.submit(process_row, row, model, args))

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
                runner_type="mistral_runner",
                prompt=prompt,
                args=args,
            )
