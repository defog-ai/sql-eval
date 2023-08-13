from eval.eval import compare_df, query_postgres_db, subset_df
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from utils.pruning import prune_metadata_str
from tqdm import tqdm
from psycopg2.extensions import QueryCanceledError
from time import time
import gc

def prepare_questions_df(questions_file, num_questions):
    question_query_df = pd.read_csv(questions_file, nrows=num_questions)
    question_query_df["generated_query"] = ""
    question_query_df["reason"] = ""
    question_query_df["error_msg"] = ""
    question_query_df["correct"] = 0
    question_query_df["subset"] = 0
    question_query_df["error_query_gen"] = 0
    question_query_df["error_db_exec"] = 0
    question_query_df["timeout"] = 0
    # add custom metrics below:
    question_query_df["latency_seconds"] = 0.0  # latency of query generation in seconds
    question_query_df["tokens_used"] = 0  # number of tokens used in query generation

    question_query_df.reset_index(inplace=True, drop=True)
    return question_query_df


def generate_prompt(prompt_file, question, db_name):
    with open(prompt_file, "r") as f:
        prompt = f.read()

    pruned_metadata_str = prune_metadata_str(question, db_name)
    prompt = prompt.format(
        user_question=question, table_metadata_string=pruned_metadata_str
    )
    return prompt


def get_tokenizer_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_cache=True,
    )
    return tokenizer, model


def run_hf_eval(
    questions_file: str,
    prompt_file: str,
    num_questions: int = None,
    model_name: str = "defog/starcoder-finetune-v3",
    output_file: str = "results.csv",
):
    print("preparing questions...")
    # get questions
    df = prepare_questions_df(questions_file, num_questions)

    # create a prompt for each question
    df["prompt"] = df[["question", "db_name"]].apply(
        lambda row: generate_prompt(prompt_file, row["question"], row["db_name"]),
        axis=1,
    )

    print("questions prepared\nnow loading model...")
    # initialize tokenizer and model
    tokenizer, model = get_tokenizer_model(model_name)

    print("model loaded\nnow generating and evaluating predictions...")
    # generate predictions
    eos_token_id = tokenizer.convert_tokens_to_ids(["```"])[0]
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # from here, just do the usual eval stuff
    total_tried = 0
    total_correct = 0
    output_rows = []
    with tqdm(total=len(df)) as pbar:
        for row in df.to_dict("records"):
            total_tried += 1
            start_time = time()
            generated_query = (
                pipe(
                    row["prompt"],
                    max_new_tokens=300,
                    do_sample=False,
                    # top_p=0.7,
                    # temperature=0.2,
                    num_beams=5,
                    num_return_sequences=1,
                    eos_token_id=eos_token_id,
                    pad_token_id=eos_token_id,
                )[0]["generated_text"]
                .split("```sql")[-1]
                .split(";")[0]
                .strip()
            )
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            end_time = time()

            row["generated_query"] = generated_query
            row["latency_seconds"] = end_time - start_time
            golden_query = row["query"]
            db_name = row["db_name"]
            question = row["question"]
            query_category = row["query_category"]
            correct = subset = 0
            generated_result = expected_result = None

            try:
                expected_result = query_postgres_db(golden_query, db_name).rename(
                    columns=str.lower
                )

                generated_result = query_postgres_db(generated_query, db_name).rename(
                    columns=str.lower
                )

                correct = subset = int(
                    compare_df(
                        expected_result, generated_result, query_category, question
                    )
                )
                if not correct:
                    subset = subset_df(
                        df_sub=expected_result,
                        df_super=generated_result,
                        query_category=query_category,
                        question=question,
                    )
                row["correct"] = int(correct)
                row["subset"] = int(subset)
                row["error_msg"] = ""
                if subset:
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

    output_df = pd.DataFrame(output_rows)
    del output_df["prompt"]
    print(output_df.groupby("query_category")[['correct', 'subset']].mean())
    output_df = output_df.sort_values(by=["db_name", "query_category", "question"])
    output_df.to_csv(output_file, index=False, float_format="%.2f")
