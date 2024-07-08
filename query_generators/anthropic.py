import time
from typing import Dict, List
from func_timeout import FunctionTimedOut, func_timeout
from anthropic import Anthropic
import os

from query_generators.query_generator import QueryGenerator
from utils.pruning import prune_metadata_str
from utils.gen_prompt import to_prompt_schema
from utils.dialects import (
    ddl_to_bigquery,
    ddl_to_mysql,
    ddl_to_sqlite,
    ddl_to_tsql,
)

anthropic = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def convert_ddl(postgres_ddl: str, to_dialect: str, db_name: str):
    if to_dialect == "postgres":
        return postgres_ddl
    elif to_dialect == "bigquery":
        new_ddl, _ = ddl_to_bigquery(postgres_ddl, "postgres", db_name, 42)
    elif to_dialect == "mysql":
        new_ddl, _ = ddl_to_mysql(postgres_ddl, "postgres", db_name, 42)
    elif to_dialect == "sqlite":
        new_ddl, _ = ddl_to_sqlite(postgres_ddl, "postgres", db_name, 42)
    elif to_dialect == "tsql":
        new_ddl, _ = ddl_to_tsql(postgres_ddl, "postgres", db_name, 42)
    else:
        raise ValueError(f"Unsupported dialect {to_dialect}")
    return new_ddl


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
        """Get Anthropic chat completion for a given prompt and model"""
        generated_text = ""
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        try:
            completion = anthropic.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
            )
            generated_text = completion.content[0].text
        except Exception as e:
            print(str(e))
            if self.verbose:
                print(type(e), e)
        return generated_text

    @staticmethod
    def count_tokens(prompt: str = "") -> int:
        """
        This function counts the number of tokens used in a prompt
        model: the model used to generate the prompt. can be one of the following: gpt-3.5-turbo-0613, gpt-4-0613, text-davinci-003
        messages: (only for OpenAI chat models) a list of messages to be used as a prompt. Each message is a dict with two keys: role and content
        prompt: (only for text-davinci-003 model) a string to be used as a prompt
        """
        num_tokens = anthropic.count_tokens(prompt)
        return num_tokens

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

        if self.use_public_data:
            from defog_data.metadata import dbs
            import defog_data.supplementary as sup
        else:
            # raise Exception("Replace this with your private data import")
            from defog_data_private.metadata import dbs

        with open(self.prompt_file) as file:
            model_prompt = file.read()
        question_instructions = question + " " + instructions
        if table_metadata_string == "":
            if columns_to_keep > 0:
                pruned_metadata_ddl, join_str = prune_metadata_str(
                    question_instructions,
                    self.db_name,
                    self.use_public_data,
                    columns_to_keep,
                    shuffle,
                )
                pruned_metadata_str = pruned_metadata_ddl + join_str
            elif columns_to_keep == 0:
                md = dbs[self.db_name]["table_metadata"]
                pruned_metadata_str = to_prompt_schema(md, shuffle)
                table_metadata_ddl = convert_ddl(
                    postgres_ddl=table_metadata_ddl,
                    to_dialect=self.db_type,
                    db_name=self.db_name,
                )
                column_join = sup.columns_join.get(self.db_name, {})
                # get join_str from column_join
                join_list = []
                for values in column_join.values():
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
                prune_metadata_str = prune_metadata_str + join_str
            else:
                raise ValueError("columns_to_keep must be >= 0")
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

        tokens_used = self.count_tokens(prompt=prompt)

        return {
            "query": self.query,
            "reason": self.reason,
            "err": self.err,
            "latency_seconds": time.time() - start_time,
            "tokens_used": tokens_used,
        }
