import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import os
from eval.eval import compare_query_results
import pandas as pd
import pandas_gbq
from psycopg2.extensions import QueryCanceledError
from query_generators.anthropic import AnthropicQueryGenerator
from tqdm import tqdm
from utils.questions import prepare_questions_df
from utils.creds import db_creds_all, bq_project
from utils.reporting import upload_results


def run_anthropic_eval(args):
    # get questions
    print("Preparing questions...")
    print(
        f"Using {'all' if args.num_questions is None else args.num_questions} question(s) from {args.questions_file}"
    )
    question_query_df = prepare_questions_df(
        args.questions_file, args.db_type, args.num_questions, args.k_shot
    )
    for prompt_file, output_file in zip(args.prompt_file, args.output_file):
        qg_class = AnthropicQueryGenerator

        input_rows = question_query_df.to_dict("records")
        output_rows = []
        with ThreadPoolExecutor(args.parallel_threads) as executor:
            # for each query in the csv, generate a query using the generator asynchronously
            futures = []
            for row in input_rows:
                # get db creds for each row's db_name
                db_name = row["db_name"]
                db_creds = db_creds_all[row["db_type"]]

                qg = qg_class(
                    db_creds=copy.deepcopy(db_creds),
                    db_name=db_name,
                    model=args.model,
                    prompt_file=prompt_file,
                    timeout=args.timeout_gen,
                    use_public_data=not args.use_private_data,
                    verbose=args.verbose,
                    instructions=row["instructions"],
                    k_shot_prompt=row["k_shot_prompt"],
                )

                generated_query_fut = executor.submit(
                    qg.generate_query,
                    question=row["question"],
                    instructions=row["instructions"],
                    k_shot_prompt=row["k_shot_prompt"],
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
                    db_type = row["db_type"]
                    question = row["question"]
                    query_category = row["query_category"]
                    exact_match = correct = 0
                    db_creds = db_creds_all[row["db_type"]]
                    # try executing the queries and compare the results if they succeed
                    try:
                        exact_match, correct = compare_query_results(
                            query_gold=expected_query,
                            query_gen=query_gen,
                            db_name=db_name,
                            db_type=db_type,
                            db_creds=db_creds_all[db_type],
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
        # get directory of output_file and create if not exist
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_df.to_csv(output_file, index=False, float_format="%.2f")

        # get average rate of exact matches
        avg_acc = output_df["exact_match"].sum() / len(output_df)
        print(f"Average rate of exact match: {avg_acc:.2f}")
        # get average rate of correct results
        avg_subset = output_df["correct"].sum() / len(output_df)
        print(f"Average correct rate: {avg_subset:.2f}")

        # save to BQ
        if args.bq_table is not None:
            run_name = output_file.split("/")[-1].split(".")[0]
            output_df["run_name"] = run_name
            output_df["run_time"] = pd.Timestamp.now()
            output_df["run_params"] = json.dumps(vars(args))
            print(f"Saving to BQ table {args.bq_table} with run_name {run_name}")
            try:
                if bq_project is not None and bq_project != "":
                    pandas_gbq.to_gbq(
                        dataframe=output_df,
                        destination_table=args.bq_table,
                        project_id=bq_project,
                        if_exists="append",
                        progress_bar=False,
                    )
                    print(f"Saved to BQ table {args.bq_table} with run_name {run_name}")
                else:
                    print("No BQ project id specified, skipping save to BQ")
            except Exception as e:
                print(f"Error saving to BQ: {e}")

        results = output_df.to_dict("records")
        # upload results
        with open(prompt_file, "r") as f:
            prompt = f.read()
        if args.upload_url is not None:
            upload_results(
                results=results,
                url=args.upload_url,
                runner_type="anthropic",
                prompt=prompt,
                args=args,
            )
