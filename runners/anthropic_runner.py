import os
from time import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import sqlparse
from tqdm import tqdm

from eval.eval import compare_query_results
from utils.creds import db_creds_all
from utils.dialects import convert_postgres_ddl_to_dialect
from utils.gen_prompt import to_prompt_schema
from utils.questions import prepare_questions_df
from utils.reporting import upload_results
from utils.llm import chat_anthropic


def generate_prompt(
    prompt_file,
    question,
    db_name,
    db_type,
    instructions="",
    k_shot_prompt="",
    glossary="",
    table_metadata_string="",
    prev_invalid_sql="",
    prev_error_msg="",
    public_data=True,
    shuffle=True,
):
    if public_data:
        from defog_data.metadata import dbs
        import defog_data.supplementary as sup
    else:
        from defog_data_private.metadata import dbs
        import defog_data_private.supplementary as sup

    with open(prompt_file, "r") as f:
        prompt = f.read()

    if table_metadata_string == "":
        md = dbs[db_name]["table_metadata"]
        pruned_metadata_ddl = to_prompt_schema(md, shuffle)
        pruned_metadata_ddl = convert_postgres_ddl_to_dialect(
            postgres_ddl=pruned_metadata_ddl,
            to_dialect=db_type,
            db_name=db_name,
        )
        column_join = sup.columns_join.get(db_name, {})
        join_list = []
        for values in column_join.values():
            if isinstance(values[0], tuple):
                for col_pair in values:
                    col_1, col_2 = col_pair
                    join_str = f"{col_1} can be joined with {col_2}"
                    if join_str not in join_list:
                        join_list.append(join_str)
            else:
                col_1, col_2 = values[0]
                join_str = f"{col_1} can be joined with {col_2}"
                if join_str not in join_list:
                    join_list.append(join_str)
        if len(join_list) > 0:
            join_str = "\nHere is a list of joinable columns:\n" + "\n".join(join_list)
        else:
            join_str = ""
        pruned_metadata_str = pruned_metadata_ddl + join_str
    else:
        pruned_metadata_str = table_metadata_string

    prompt = prompt.format(
        user_question=question,
        db_type=db_type,
        instructions=instructions,
        table_metadata_string=pruned_metadata_str,
        k_shot_prompt=k_shot_prompt,
        glossary=glossary,
        prev_invalid_sql=prev_invalid_sql,
        prev_error_msg=prev_error_msg,
    )
    return prompt


def process_row(row, model_name, args):
    start_time = time()
    prompt = generate_prompt(
        prompt_file=args.prompt_file[0],
        question=row["question"],
        db_name=row["db_name"],
        db_type=args.db_type,
        instructions=row["instructions"],
        k_shot_prompt=row["k_shot_prompt"],
        glossary=row["glossary"],
        table_metadata_string=row["table_metadata_string"],
        prev_invalid_sql=row["prev_invalid_sql"],
        prev_error_msg=row["prev_error_msg"],
        public_data=not args.use_private_data,
        shuffle=args.shuffle_metadata,
    )
    messages = [{"role": "user", "content": prompt}]
    try:
        response = chat_anthropic(messages=messages, model=model_name, temperature=0.0)
        generated_query = (
            response.content.split("```sql", 1)[-1].split("```", 1)[0].strip()
        )
        try:
            generated_query = sqlparse.format(
                generated_query, reindent=True, keyword_case="upper"
            )
        except:
            pass
        return {
            "query": generated_query,
            "reason": "",
            "err": "",
            "latency_seconds": time() - start_time,
            "tokens_used": response.input_tokens + response.output_tokens,
        }
    except Exception as e:
        return {
            "query": "",
            "reason": "",
            "err": f"GENERATION ERROR: {str(e)}",
            "latency_seconds": time() - start_time,
            "tokens_used": 0,
        }


def run_anthropic_eval(args):
    # get params from args
    questions_file_list = args.questions_file
    prompt_file_list = args.prompt_file
    output_file_list = args.output_file
    num_questions = args.num_questions
    k_shot = args.k_shot
    db_type = args.db_type
    cot_table_alias = args.cot_table_alias

    for questions_file, prompt_file, output_file in zip(
        questions_file_list, prompt_file_list, output_file_list
    ):
        print(f"Using prompt file {prompt_file}")
        print("Preparing questions...")
        print(
            f"Using {'all' if num_questions is None else num_questions} question(s) from {questions_file}"
        )
        question_query_df = prepare_questions_df(
            questions_file, db_type, num_questions, k_shot, cot_table_alias
        )
        input_rows = question_query_df.to_dict("records")
        output_rows = []
        with ThreadPoolExecutor(args.parallel_threads) as executor:
            futures = []
            for row in input_rows:
                generated_query_fut = executor.submit(
                    process_row,
                    row=row,
                    model_name=args.model,
                    args=args,
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
                elif "TIMEOUT" in err:
                    row["timeout"] = 1
                else:
                    expected_query = row["query"]
                    db_name = row["db_name"]
                    db_type = row["db_type"]
                    try:
                        exact_match, correct = compare_query_results(
                            query_gold=expected_query,
                            query_gen=query_gen,
                            db_name=db_name,
                            db_type=db_type,
                            db_creds=db_creds_all[db_type],
                            question=row["question"],
                            query_category=row["query_category"],
                            decimal_points=args.decimal_points,
                        )
                        if correct:
                            total_correct += 1
                            row["is_correct"] = 1
                            row["error_msg"] = ""
                        else:
                            row["is_correct"] = 0
                            row["error_msg"] = "INCORRECT RESULTS"
                    except Exception as e:
                        row["error_db_exec"] = 1
                        row["error_msg"] = f"EXECUTION ERROR: {str(e)}"
                output_rows.append(row)
                pbar.set_description(
                    f"Accuracy: {round(total_correct/total_tried * 100, 2)}% ({total_correct}/{total_tried})"
                )

        # save results to csv
        output_df = pd.DataFrame(output_rows)
        output_df = output_df.sort_values(by=["db_name", "query_category", "question"])
        # get directory of output_file and create if not exist
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_df.to_csv(output_file, index=False, float_format="%.2f")

        # get average rate of correct results
        avg_subset = output_df["is_correct"].sum() / len(output_df)
        print(f"Average correct rate: {avg_subset:.2f}")

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
