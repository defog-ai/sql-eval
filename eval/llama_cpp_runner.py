import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from eval.eval import compare_query_results
import pandas as pd
from utils.pruning import prune_metadata_str
from utils.questions import prepare_questions_df
from utils.creds import db_creds_all
from tqdm import tqdm
from time import time
from utils.reporting import upload_results
from llama_cpp import Llama


def generate_prompt(
    prompt_file, question, db_name, instructions="", k_shot_prompt="", public_data=True
):
    with open(prompt_file, "r") as f:
        prompt = f.read()
    question_instructions = question + " " + instructions

    pruned_metadata_str = prune_metadata_str(
        question_instructions, db_name, public_data
    )
    prompt = prompt.format(
        user_question=question,
        instructions=instructions,
        table_metadata_string=pruned_metadata_str,
        k_shot_prompt=k_shot_prompt,
    )
    return prompt


def process_row(llm, row):
    start_time = time()
    prompt = row["prompt"]
    generated_query = (
        llm(
            prompt,
            max_tokens=512,
            temperature=0,
            top_p=1,
            echo=False,
            repeat_penalty=1.0,
        )["choices"][0]["text"]
        .split(";")[0]
        .split("```")[0]
        .strip()
        + ";"
    )
    end_time = time()
    row["generated_query"] = generated_query
    row["latency_seconds"] = end_time - start_time
    golden_query = row["query"]
    db_name = row["db_name"]
    db_type = row["db_type"]
    question = row["question"]
    query_category = row["query_category"]
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
        )
        row["exact_match"] = int(exact_match)
        row["correct"] = int(correct)
        row["error_msg"] = ""
    except Exception as e:
        row["error_db_exec"] = 1
        row["error_msg"] = f"QUERY EXECUTION ERROR: {e}"
    return row


def run_llama_cpp_eval(args):
    # get params from args
    questions_file = args.questions_file
    prompt_file_list = args.prompt_file
    num_questions = args.num_questions
    public_data = not args.use_private_data
    model_path = args.model
    output_file_list = args.output_file
    k_shot = args.k_shot
    max_workers = args.parallel_threads
    db_type = args.db_type

    llm = Llama(model_path=model_path, n_gpu_layers=1, n_ctx=2048)

    # get questions
    print("Preparing questions...")
    print(
        f"Using {'all' if num_questions is None else num_questions} question(s) from {questions_file}"
    )
    df = prepare_questions_df(questions_file, db_type, num_questions, k_shot)

    for prompt_file, output_file in zip(prompt_file_list, output_file_list):
        # create a prompt for each question
        df["prompt"] = df[
            ["question", "db_name", "instructions", "k_shot_prompt"]
        ].apply(
            lambda row: generate_prompt(
                prompt_file,
                row["question"],
                row["db_name"],
                row["instructions"],
                row["k_shot_prompt"],
                public_data,
            ),
            axis=1,
        )

        total_tried = 0
        total_correct = 0
        output_rows = []

        with tqdm(total=len(df)) as pbar:
            for row in df.to_dict("records"):
                row = process_row(llm, row)
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
        print(output_df.groupby("query_category")[["exact_match", "correct"]].mean())
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
