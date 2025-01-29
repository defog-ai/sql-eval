from typing import Dict, List
import time
from func_timeout import FunctionTimedOut, func_timeout
import json

from query_generators.query_generator import QueryGenerator
from utils.gen_prompt import to_prompt_schema
from utils.dialects import convert_postgres_ddl_to_dialect
from utils.llm import chat_openai, LLMResponse


class OpenAIQueryGenerator(QueryGenerator):
    """
    Query generator that uses OpenAI's models
    """

    def __init__(
        self,
        db_creds: Dict[str, str],
        db_name: str,
        db_type: str,
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
        self.o1 = self.model.startswith("o1-")
        self.prompt_file = prompt_file
        self.use_public_data = use_public_data
        self.timeout = timeout
        self.verbose = verbose

    def get_chat_completion(
        self,
        model,
        messages,
        max_tokens=600,
        temperature=0,
        stop=[],
        logit_bias={},
        seed=100,
    ) -> str:
        """Get OpenAI chat completion using the new utility function"""
        try:
            response: LLMResponse = chat_openai(
                messages=messages,
                model=model,
                max_completion_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                seed=seed,
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
        table_aliases: str,
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
            chat_prompt = json.load(file)
        question_instructions = question + " " + instructions

        if table_metadata_string == "":
            md = dbs[self.db_name]["table_metadata"]
            table_metadata_ddl = to_prompt_schema(md, shuffle)
            table_metadata_ddl = convert_postgres_ddl_to_dialect(
                postgres_ddl=table_metadata_ddl,
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
            table_metadata_string = table_metadata_ddl + join_str

        if glossary == "":
            glossary = dbs[self.db_name]["glossary"]
        try:
            if self.o1:
                sys_prompt = ""
                user_prompt = chat_prompt[0]["content"]
            else:
                sys_prompt = chat_prompt[0]["content"]
                sys_prompt = sys_prompt.format(
                    db_type=self.db_type,
                )
                user_prompt = chat_prompt[1]["content"]
                if len(chat_prompt) == 3:
                    assistant_prompt = chat_prompt[2]["content"]
        except:
            raise ValueError("Invalid prompt file. Please use prompt_openai.md")
        user_prompt = user_prompt.format(
            db_type=self.db_type,
            user_question=question,
            table_metadata_string=table_metadata_string,
            instructions=instructions,
            k_shot_prompt=k_shot_prompt,
            glossary=glossary,
            prev_invalid_sql=prev_invalid_sql,
            prev_error_msg=prev_error_msg,
            table_aliases=table_aliases,
        )

        if self.o1:
            messages = [{"role": "user", "content": user_prompt}]
        else:
            messages = []
            messages.append({"role": "system", "content": sys_prompt})
            messages.append({"role": "user", "content": user_prompt})
            if len(chat_prompt) == 3:
                messages.append({"role": "assistant", "content": assistant_prompt})

        function_to_run = self.get_chat_completion
        package = messages

        try:
            self.completion = func_timeout(
                self.timeout,
                function_to_run,
                args=(self.model, package, 1200, 0),
            )
            results = self.completion
            self.query = results.split("```sql")[-1].split("```")[0]
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
            print(e)
            if isinstance(e, KeyError):
                self.err = f"QUERY GENERATION ERROR: {type(e)}, {e}, Completion: {self.completion}"
            else:
                self.err = f"QUERY GENERATION ERROR: {type(e)}, {e}"

        return {
            "table_metadata_string": table_metadata_string,
            "query": self.query,
            "reason": self.reason,
            "err": self.err,
            "latency_seconds": time.time() - start_time,
            "tokens_used": tokens_used,
        }
