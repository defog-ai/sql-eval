import json
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


def generate_base_prompt(
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
    """
    Base prompt generation logic used by all runners.
    """
    if public_data:
        from defog_data.metadata import dbs
        import defog_data.supplementary as sup
    else:
        from defog_data_private.metadata import dbs
        import defog_data_private.supplementary as sup

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
        join_str = (
            "\nHere is a list of joinable columns:\n" + "\n".join(join_list)
            if join_list
            else ""
        )
        pruned_metadata_str = pruned_metadata_ddl + join_str
    else:
        pruned_metadata_str = table_metadata_string

    return {
        "prompt_file": prompt_file,
        "question": question,
        "db_type": db_type,
        "instructions": instructions,
        "table_metadata_string": pruned_metadata_str,
        "k_shot_prompt": k_shot_prompt,
        "glossary": glossary,
        "prev_invalid_sql": prev_invalid_sql,
        "prev_error_msg": prev_error_msg,
    }


def extract_sql_from_response(content):
    """Extract SQL from between ```sql blocks and format it."""
    try:
        generated_query = content.split("```sql", 1)[-1].split("```", 1)[0].strip()
        return sqlparse.format(generated_query, reindent=True, keyword_case="upper")
    except:
        return content


def run_eval_in_threadpool(df, model_name, process_row_func, args):
    """Common threadpool execution pattern for all runners."""
    total_tried = 0
    total_correct = 0
    output_rows = []

    print(f"Running evaluation using {model_name}...")
    with ThreadPoolExecutor(max_workers=args.parallel_threads) as executor:
        futures = []
        for row in df.to_dict("records"):
            futures.append(executor.submit(process_row_func, row, model_name, args))

        with tqdm(as_completed(futures), total=len(futures)) as pbar:
            for f in pbar:
                row = f.result()
                output_rows.append(row)
                if row.get("correct", 0):
                    total_correct += 1
                total_tried += 1
                pbar.set_description(
                    f"Acc: {total_correct}/{total_tried}={total_correct/total_tried:.3f}"
                )

    return output_rows, total_correct, total_tried
