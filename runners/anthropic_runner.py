from time import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

import pandas as pd
import sqlparse
from tqdm import tqdm

from runners.base_runner import run_eval_in_threadpool
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
    try:
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
    except ImportError:
        # When defog_data is not available, just format with the existing table_metadata_string
        with open(prompt_file, "r") as f:
            prompt = f.read()
        
        prompt = prompt.format(
            user_question=question,
            db_type=db_type,
            instructions=instructions,
            table_metadata_string=table_metadata_string,
            k_shot_prompt=k_shot_prompt,
            glossary=glossary,
            prev_invalid_sql=prev_invalid_sql,
            prev_error_msg=prev_error_msg,
        )
        return prompt


def process_row(row, model_name, args):
    start_time = time()
    result_row = row.copy()  # Create a copy of the original row to maintain all data
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
        result_row.update({
            "generated_query": generated_query,
            "reason": "",
            "error_msg": "",
            "latency_seconds": time() - start_time,
            "tokens_used": response.input_tokens + response.output_tokens,
        })
        
        # Verify the generated query
        try:
            exact_match, correct = compare_query_results(
                query_gold=row["query"],
                query_gen=generated_query,
                db_name=row["db_name"],
                db_type=args.db_type,
                db_creds=db_creds_all[args.db_type],
                question=row["question"],
                query_category=row["query_category"],
                decimal_points=args.decimal_points if hasattr(args, 'decimal_points') else 2,
            )
            result_row["exact_match"] = int(exact_match)
            result_row["correct"] = int(correct)
            result_row["is_correct"] = int(correct)
        except Exception as e:
            result_row["error_db_exec"] = 1
            result_row["error_msg"] = f"EXECUTION ERROR: {str(e)}"
            result_row["is_correct"] = 0
    except Exception as e:
        result_row.update({
            "generated_query": "",
            "reason": "",
            "error_msg": f"GENERATION ERROR: {str(e)}",
            "latency_seconds": time() - start_time,
            "tokens_used": 0,
            "is_correct": 0,
        })
    return result_row


def run_anthropic_eval(args):
    """Run evaluation using Anthropic"""
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
        df = prepare_questions_df(
            questions_file, db_type, num_questions, k_shot, cot_table_alias
        )

        output_rows, total_correct, total_tried = run_eval_in_threadpool(
            df, args.model, process_row, args
        )

        # Convert to DataFrame and save results
        output_df = pd.DataFrame(output_rows)
        output_df = output_df.sort_values(by=["db_name", "query_category", "question"])
        if "prompt" in output_df.columns:
            del output_df["prompt"]

        # Get stats by query category
        agg_stats = (
            output_df.groupby("query_category")
            .agg(
                num_rows=("db_name", "count"),
                mean_correct=("is_correct", "mean"),
                mean_error_db_exec=("error_db_exec", "mean"),
            )
            .reset_index()
        )
        print(agg_stats)

        # Create output directory if needed
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_df.to_csv(output_file, index=False, float_format="%.2f")

        # Print summary stats
        print(f"Total questions: {total_tried}")
        print(f"Total correct: {total_correct}")
        print(f"Accuracy: {total_correct/total_tried:.3f}")

        # Upload results if URL provided
        try:
            if hasattr(args, "upload_url") and args.upload_url:
                with open(prompt_file, "r") as f:
                    prompt = f.read()
                upload_results(
                    results=output_df.to_dict("records"),
                    url=args.upload_url,
                    runner_type="anthropic",
                    prompt=prompt,
                    args=args,
                )
        except Exception as e:
            print(f"Error uploading results: {e}")
