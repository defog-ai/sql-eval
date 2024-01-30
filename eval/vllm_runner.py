import json
import os
import sqlparse
from vllm import LLM, SamplingParams
from eval.eval import compare_query_results
import pandas as pd
from utils.pruning import prune_metadata_str
from utils.questions import prepare_questions_df
from utils.creds import db_creds_all, bq_project
import time
import torch
from transformers import AutoTokenizer
from tqdm import tqdm


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


def run_vllm_eval(args):
    # get params from args
    questions_file = args.questions_file
    prompt_file_list = args.prompt_file
    num_questions = args.num_questions
    public_data = not args.use_private_data
    model_name = args.model
    output_file_list = args.output_file
    num_beams = args.num_beams
    k_shot = args.k_shot
    db_type = args.db_type

    # initialize model only once as it takes a while
    print(f"Preparing {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(model=model_name, tensor_parallel_size=torch.cuda.device_count())

    sampling_params = SamplingParams(
        n=1,
        best_of=num_beams,
        use_beam_search=num_beams != 1,
        stop_token_ids=[tokenizer.eos_token_id],
        max_tokens=300,
        temperature=0,
    )
    # get questions
    print("Preparing questions...")
    print(
        f"Using {'all' if num_questions is None else num_questions} question(s) from {questions_file}"
    )
    df = prepare_questions_df(questions_file, db_type, num_questions, k_shot)

    for prompt_file, output_file in zip(prompt_file_list, output_file_list):
        print(f"Using prompt file {prompt_file}")
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
        print(f"Prepared {len(df)} question(s) from {questions_file}")
        print(f"Generating completions")
        start_time = time.time()
        # we pass the full list of prompts at once to the vllm engine
        outputs = llm.generate(df["prompt"].tolist(), sampling_params)
        time_taken = time.time() - start_time
        print(f"Time taken: {time_taken:.1f}s")

        # save generation metrics
        df["latency_seconds"] = time_taken / len(df)

        df["generated_query"] = ""
        df["tokens_used"] = 0
        df["correct"] = 0
        df["exact_match"] = 0
        df["error_db_exec"] = 0
        df["error_msg"] = ""
        total_correct = 0
        with tqdm(total=len(df)) as pbar:
            for i, output in enumerate(outputs):
                generated_query = output.outputs[0].text.split(";")[0].split("```")[0].strip()
                normalized_query = sqlparse.format(
                    generated_query, keyword_case="upper", strip_whitespace=True
                )
                df.loc[i, "generated_query"] = normalized_query
                df.loc[i, "tokens_used"] = len(output.outputs[0].token_ids)
                df.loc[i, "latency_seconds"] = time_taken / len(df)
                row = df.iloc[i]
                golden_query = row["query"]
                db_name = row["db_name"]
                db_type = row["db_type"]
                question = row["question"]
                query_category = row["query_category"]
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
                    )
                    df.loc[i, "exact_match"] = int(exact_match)
                    df.loc[i, "correct"] = int(correct)
                    df.loc[i, "error_msg"] = ""
                    if correct:
                        total_correct += 1
                except Exception as e:
                    df.loc[i, "error_db_exec"] = 1
                    df.loc[i, "error_msg"] = f"QUERY EXECUTION ERROR: {e}"
                pbar.update(1)
                pbar.set_description(
                    f"Correct so far: {total_correct}/{(i+1)} ({100*total_correct/(i+1):.2f}%)"
                )
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

        # save to BQ
        if args.bq_table is not None:
            run_name = output_file.split("/")[-1].split(".")[0]
            df["run_name"] = run_name
            df["run_time"] = pd.Timestamp.now()
            df["run_params"] = json.dumps(vars(args))
            print(f"Saving to BQ table {args.bq_table} with run_name {run_name}")
            try:
                if bq_project is not None and bq_project != "":
                    df.to_gbq(
                        destination_table=args.bq_table,
                        project_id=bq_project,
                        if_exists="append",
                        progress_bar=False,
                    )
                else:
                    print("No BQ project id specified, skipping save to BQ")
            except Exception as e:
                print(f"Error saving to BQ: {e}")
