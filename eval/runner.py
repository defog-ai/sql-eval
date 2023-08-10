from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from eval.eval import compare_df, query_postgres_db, subset_df
import pandas as pd
from psycopg2.extensions import QueryCanceledError
from query_generators.openai import OpenAIChatQueryGenerator
from tqdm import tqdm


def run(args):
    question_query_df = pd.read_csv(args.questions_file, nrows=args.num_questions)
    if args.qg_class == "oa_chat":
        qg_class = OpenAIChatQueryGenerator
    else:
        raise ValueError(f"Unknown qg_class {args.qg_class}")
    # add columns for generated query and metrics
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
                correct = subset = 0
                generated_result = expected_result = None
                db_creds = {
                    "host": "localhost",
                    "port": 5432,
                    "user": "postgres",
                    "password": "postgres",
                    "database": db_name,
                }
                # try executing the queries and compare the results if they succeed
                try:
                    expected_result = query_postgres_db(
                        expected_query, db_name, db_creds, args.timeout_exec
                    )
                    expected_result = expected_result.rename(columns=str.lower)
                    generated_result = query_postgres_db(
                        query_gen, db_name, db_creds, args.timeout_exec
                    )
                    generated_result = generated_result.rename(columns=str.lower)
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
                            verbose=args.verbose,
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
            pbar.set_description(
                f"Correct so far: {total_correct}/{total_tried} ({100*total_correct/total_tried:.2f}%)"
            )
    output_df = pd.DataFrame(output_rows)
    output_df = output_df.sort_values(by=["db_name", "query_category", "question"])
    output_df.to_csv(args.output_file, index=False, float_format="%.2f")

    # get average accuracy
    avg_acc = output_df["correct"].sum() / len(output_df)
    print(f"Average accuracy: {avg_acc:.2f}")
    # get average subset or correct accuracy
    avg_subset = output_df["subset"].sum() / len(output_df)
    print(f"Average subset accuracy: {avg_subset:.2f}")
