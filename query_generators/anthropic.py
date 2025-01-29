from typing import Dict
from func_timeout import FunctionTimedOut, func_timeout
import os
import time

from query_generators.query_generator import QueryGenerator
from utils.gen_prompt import to_prompt_schema
from utils.dialects import convert_postgres_ddl_to_dialect
from utils.llm import chat_anthropic


class AnthropicQueryGenerator(QueryGenerator):
    """
    Query generator that uses Anthropic's models
    Models available: claude-2, claude-instant-1
    """

    def __init__(
        self,
        db_type: str,
        db_creds: Dict[str, str],
        db_name: str,
        model: str,
        prompt_file: str,
        timeout: int,
        use_public_data: bool,
        verbose: bool,
        **kwargs,
    ):
        self.db_creds = db_creds
        self.db_type = db_type
        self.db_name = db_name
        self.model = model
        self.prompt_file = prompt_file
        self.use_public_data = use_public_data
        self.timeout = timeout
        self.verbose = verbose

    def get_completion(
        self,
        model,
        prompt,
        max_tokens=600,
        temperature=0,
        stop=["```", ";"],
        logit_bias={},
    ):
        """Get Anthropic chat completion using the new utility function"""
        messages = [{"role": "user", "content": prompt}]
        try:
            response = chat_anthropic(
                messages=messages,
                model=model,
                max_completion_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
            )
            return response.content
        except Exception as e:
            print(str(e))
            if self.verbose:
                print(type(e), e)
            return ""

    def generate_query(
        self,
        question: str,
        instructions: str,
        k_shot_prompt: str,
        glossary: str,
        table_metadata_string: str,
        prev_invalid_sql: str,
        prev_error_msg: str,
        cot_instructions: str,
        columns_to_keep: int,
        shuffle: bool,
    ) -> dict:
        start_time = time.time()
        self.err = ""
        self.query = ""
        self.reason = ""
        tokens_used = 0

        if self.use_public_data:
            from defog_data.metadata import dbs
            import defog_data.supplementary as sup
        else:
            # raise Exception("Replace this with your private data import")
            from defog_data_private.metadata import dbs
            import defog_data_private.supplementary as sup

        with open(self.prompt_file) as file:
            model_prompt = file.read()
        question_instructions = question + " " + instructions

        if table_metadata_string == "":
            md = dbs[self.db_name]["table_metadata"]
            pruned_metadata_str = to_prompt_schema(md, shuffle)
            pruned_metadata_str = convert_postgres_ddl_to_dialect(
                postgres_ddl=pruned_metadata_str,
                to_dialect=self.db_type,
                db_name=self.db_name,
            )
            column_join = sup.columns_join.get(self.db_name, {})
            # get join_str from column_join
            join_list = []
            for values in column_join.values():
                if isinstance(values[0], tuple):
                    for col_pair in values:
                        col_1, col_2 = col_pair
                        # add to join_list
                        join_str = f"{col_1} can be joined with {col_2}"
                        if join_str not in join_list:
                            join_list.append(join_str)
                else:
                    col_1, col_2 = values[0]
                    # add to join_list
                    join_str = f"{col_1} can be joined with {col_2}"
                    if join_str not in join_list:
                        join_list.append(join_str)
            if len(join_list) > 0:
                join_str = "\nHere is a list of joinable columns:\n" + "\n".join(
                    join_list
                )
            else:
                join_str = ""
            pruned_metadata_str = pruned_metadata_str + join_str
        else:
            pruned_metadata_str = table_metadata_string

        prompt = model_prompt.format(
            user_question=question,
            db_type=self.db_type,
            table_metadata_string=pruned_metadata_str,
            instructions=instructions,
            k_shot_prompt=k_shot_prompt,
            glossary=glossary,
            prev_invalid_sql=prev_invalid_sql,
            prev_error_msg=prev_error_msg,
            cot_instructions=cot_instructions,
        )
        function_to_run = self.get_completion
        package = prompt

        try:
            self.completion = func_timeout(
                self.timeout,
                function_to_run,
                args=(
                    self.model,
                    package,
                    600,
                    0,
                    ["```", ";"],
                ),
            )
            results = self.completion
            self.query = results.split("```sql")[-1].split(";")[0].split("```")[0]
            self.reason = "-"
        except FunctionTimedOut:
            if self.verbose:
                print("generating query timed out")
            self.err = "QUERY GENERATION TIMEOUT"
        except Exception as e:
            if self.verbose:
                print(f"Error while generating query: {type(e)}, {e})")
            self.query = ""
            self.reason = ""
            if isinstance(e, KeyError):
                self.err = f"QUERY GENERATION ERROR: {type(e)}, {e}, Completion: {self.completion}"
            else:
                self.err = f"QUERY GENERATION ERROR: {type(e)}, {e}"

        return {
            "query": self.query,
            "reason": self.reason,
            "err": self.err,
            "latency_seconds": time.time() - start_time,
            "tokens_used": tokens_used,
            "table_metadata_string": pruned_metadata_str,
        }
