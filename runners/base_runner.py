import os
from time import time
import pandas as pd
import sqlparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from eval.eval import compare_query_results
from utils.creds import db_creds_all
from utils.dialects import convert_postgres_ddl_to_dialect
from utils.gen_prompt import to_prompt_schema
from utils.questions import prepare_questions_df
from utils.reporting import upload_results


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

    return pruned_metadata_str


def format_sql_query(generated_query):
    try:
        return sqlparse.format(generated_query, reindent=True, keyword_case="upper")
    except:
        return generated_query


class BaseRunner:
    def __init__(self):
        pass

    def _load_prompt(self, prompt_file):
        """Load prompt from file. Override in subclass if format differs."""
        with open(prompt_file, "r") as f:
            return f.read()

    def _format_prompt(self, prompt, **kwargs):
        """Format the prompt with variables. Override in subclass if format differs."""
        return prompt.format(**kwargs)

    def _call_llm(self, messages, model_name, temperature=0.0):
        """Call LLM API. Must be implemented in subclass."""
        raise NotImplementedError("Subclass must implement _call_llm")

    def _extract_query(self, response_content):
        """Extract SQL query from response. Override in subclass if format differs."""
        return response_content.split("```sql", 1)[-1].split("```", 1)[0].strip()

    def process_row(self, row, model_name, args):
        start_time = time()
        try:
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
                shuffle_metadata=args.shuffle_metadata,
            )

            response = self._call_llm(prompt, model_name)
            generated_query = self._extract_query(response.content)
            generated_query = format_sql_query(generated_query)

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

    def run_eval(self, args):
        """Common evaluation logic."""
        questions_file_list = args.questions_file
        prompt_file_list = args.prompt_file
        output_file_list = args.output_file
        model_name = args.model

        for questions_file, prompt_file, output_file in zip(
            questions_file_list, prompt_file_list, output_file_list
        ):
            print(f"Using prompt file {prompt_file}")
            print("Preparing questions...")
            question_query_df = prepare_questions_df(
                questions_file,
                args.db_type,
                args.num_questions,
                args.k_shot,
                args.cot_table_alias,
            )
            input_rows = question_query_df.to_dict("records")

            results = []
            with ThreadPoolExecutor(max_workers=args.parallel_threads) as executor:
                futures = [
                    executor.submit(self.process_row, row, model_name, args)
                    for row in input_rows
                ]
                for future in tqdm(
                    as_completed(futures), total=len(futures), desc="Processing"
                ):
                    result = future.result()
                    results.append(result)

            results_df = pd.DataFrame(results)
            results_df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")
