from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from eval.eval import compare_query_results
import pandas as pd
from psycopg2.extensions import QueryCanceledError
from query_generators.openai import OpenAIQueryGenerator
from tqdm import tqdm
from utils.questions import prepare_questions_df


def run_openai_eval(args):
    print("preparing questions...")
    # get questions
    question_query_df = prepare_questions_df(args.questions_file, args.num_questions)
    qg_class = OpenAIQueryGenerator
    # add columns for generated query and metrics
    question_query_df["generated_query"] = ""
    question_query_df["reason"] = ""
    question_query_df["error_msg"] = ""
    question_query_df["exact_match"] = 0
    question_query_df["correct"] = 0
    question_query_df["error_query_gen"] = 0
    question_query_df["error_db_exec"] = 0
    question_query_df["timeout"] = 0
    # add custom metrics below:
    question_query_df["latency_seconds"] = 0.0  # latency of query generation in seconds
    question_query_df["tokens_used"] = 0  # number of tokens used in query generation

    question_query_df.reset_index(inplace=True, drop=True)

    input_rows = question_query_df.to_dict("records")
    output_rows = []
    with ThreadPoolExecutor(args.parallel_threads) as executor:
        # for each query in the csv, generate a query using the generator asynchronously
        futures = []
        for row in input_rows:
            # get db creds for each row's db_name
            db_name = row["db_name"]
            db_creds = {
                "host": "localhost",
                "port": 5432,
                "user": "postgres",
                "password": "postgres",
                "database": db_name,
            }

            qg = qg_class(
                db_creds=copy.deepcopy(db_creds),
                model=args.model,
                prompt_file=args.prompt_file,
                timeout=args.timeout_gen,
                verbose=args.verbose,
            )

            generated_query_fut = executor.submit(
                qg.generate_query, question=row["question"]
            )
            futures.append(generated_query_fut)

        total_tried = 0
        total_correct = 0
        for f in (pbar := tqdm(as_completed(futures), total=len(futures))):
            total_tried += 1
            i = futures.index(f)
            row = input_rows[i]
            result_dict = f.result()
            query_gen = result_dict["query"]
            reason = result_dict["reason"]
            err = result_dict["err"]
            # save custom metrics
            if "latency_seconds" in result_dict:
                row["latency_seconds"] = result_dict["latency_seconds"]
            if "tokens_used" in result_dict:
                row["tokens_used"] = result_dict["tokens_used"]
            row["generated_query"] = query_gen
            row["reason"] = reason
            row["error_msg"] = err
            # save failures into relevant columns in the dataframe
            if "GENERATION ERROR" in err:
                row["error_query_gen"] = 1
            elif "EXECUTION ERROR" in err:
                row["error_db_exec"] = 1
            elif "TIMEOUT" in err:
                row["timeout"] = 1
            else:
                expected_query = row["query"]
                db_name = row["db_name"]
                question = row["question"]
                query_category = row["query_category"]
                exact_match = correct = 0
                db_creds = {
                    "host": "localhost",
                    "port": 5432,
                    "user": "postgres",
                    "password": "postgres",
                    "database": db_name,
                }
                # try executing the queries and compare the results if they succeed
                try:
                    exact_match, correct = compare_query_results(
                        query_gold=expected_query,
                        query_gen=query_gen,
                        db_name=db_name,
                        db_creds=db_creds,
                        timeout=args.timeout_exec,
                        question=question,
                        query_category=query_category,
                    )
                    row["exact_match"] = int(exact_match)
                    row["correct"] = int(correct)
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
            pbar.set_description(
                f"Correct so far: {total_correct}/{total_tried} ({100*total_correct/total_tried:.2f}%)"
            )
    output_df = pd.DataFrame(output_rows)
    output_df = output_df.sort_values(by=["db_name", "query_category", "question"])
    output_df.to_csv(args.output_file, index=False, float_format="%.2f")

    # get average rate of exact matches
    avg_acc = output_df["exact_match"].sum() / len(output_df)
    print(f"Average rate of exact match: {avg_acc:.2f}")
    # get average rate of correct results
    avg_subset = output_df["correct"].sum() / len(output_df)
    print(f"Average correct rate: {avg_subset:.2f}")
