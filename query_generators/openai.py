import time
from typing import Dict, List

from func_timeout import FunctionTimedOut, func_timeout
from openai import OpenAI
import tiktoken
import json

from query_generators.query_generator import QueryGenerator
from utils.gen_prompt import generate_prompt
from utils.dialects import (
    convert_postgres_ddl_to_dialect,
)

openai = OpenAI()


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
    ):
        """Get OpenAI chat completion for a given prompt and model"""
        generated_text = ""
        try:
            if self.o1:
                completion = openai.chat.completions.create(
                    model=model,
                    messages=messages,
                    seed=seed,
                )
            else:
                completion = openai.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
                    logit_bias=logit_bias,
                    seed=seed,
                )
            generated_text = completion.choices[0].message.content
        except Exception as e:
            print(type(e), e)
        return generated_text

    def get_nonchat_completion(
        self,
        model,
        prompt,
        max_tokens=600,
        temperature=0,
        stop=[],
        logit_bias={},
    ):
        """Get OpenAI nonchat completion for a given prompt and model"""
        generated_text = ""
        try:
            completion = openai.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                logit_bias=logit_bias,
                seed=42,
            )
            generated_text = completion["choices"][0]["text"]
        except Exception as e:
            print(type(e), e)
        return generated_text

    @staticmethod
    def count_tokens(
        model: str, messages: List[Dict[str, str]] = [], prompt: str = ""
    ) -> int:
        """
        This function counts the number of tokens used in a prompt
        model: the model used to generate the prompt. can be any valid OpenAI model
        messages: (only for OpenAI chat models) a list of messages to be used as a prompt. Each message is a dict with two keys: role and content
        """
        try:
            tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            # default to o200k_base if the model is not in the list. this is just for approximating the max token count
            tokenizer = tiktoken.get_encoding("o200k_base")
        num_tokens = 0
        for message in messages:
            for _, value in message.items():
                num_tokens += len(tokenizer.encode(value))
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
        table_aliases: str,
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
            chat_prompt = json.load(file)
        question_instructions = question + " " + instructions
        if table_metadata_string == "":
            table_metadata_string = generate_prompt(dbs[self.db_name]["table_metadata"], shuffle)
            table_metadata_string = convert_postgres_ddl_to_dialect(
                postgres_ddl=table_metadata_string,
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
            table_metadata_string = table_metadata_string + join_str
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

        function_to_run = None
        package = None
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

        tokens_used = self.count_tokens(self.model, messages=messages)

        return {
            "table_metadata_string": table_metadata_string,
            "query": self.query,
            "reason": self.reason,
            "err": self.err,
            "latency_seconds": time.time() - start_time,
            "tokens_used": tokens_used,
        }
