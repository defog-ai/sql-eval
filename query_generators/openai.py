import time
from typing import Dict, List

from func_timeout import FunctionTimedOut, func_timeout
from openai import OpenAI
import tiktoken
import json

from query_generators.query_generator import QueryGenerator
from utils.pruning import prune_metadata_str
from utils.gen_prompt import to_prompt_schema

openai = OpenAI()


class OpenAIQueryGenerator(QueryGenerator):
    """
    Query generator that uses OpenAI's models
    """

    def __init__(
        self,
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
        self.db_name = db_name
        self.model = model
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
        tokenizer = tiktoken.encoding_for_model(model)
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
        else:
            # raise Exception("Replace this with your private data import")
            from defog_data_private.metadata import dbs

        with open(self.prompt_file) as file:
            chat_prompt = json.load(file)
        question_instructions = question + " " + instructions
        if table_metadata_string == "":
            if columns_to_keep > 0:
                table_metadata_ddl, join_str = prune_metadata_str(
                    question_instructions,
                    self.db_name,
                    self.use_public_data,
                    columns_to_keep,
                    shuffle,
                )
                table_metadata_string = table_metadata_ddl + join_str
            elif columns_to_keep == 0:
                md = dbs[self.db_name]["table_metadata"]
                table_metadata_string = to_prompt_schema(md, shuffle)
            else:
                raise ValueError("columns_to_keep must be >= 0")
        if glossary == "":
            glossary = dbs[self.db_name]["glossary"]
        try:
            sys_prompt = chat_prompt[0]["content"]
            user_prompt = chat_prompt[1]["content"]
            assistant_prompt = chat_prompt[2]["content"]
        except:
            raise ValueError("Invalid prompt file. Please use prompt_openai.md")
        user_prompt = user_prompt.format(
            user_question=question,
            table_metadata_string=table_metadata_string,
            instructions=instructions,
            k_shot_prompt=k_shot_prompt,
            glossary=glossary,
            prev_invalid_sql=prev_invalid_sql,
            prev_error_msg=prev_error_msg,
            cot_instructions=cot_instructions,
        )

        messages = []
        messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": user_prompt})
        messages.append({"role": "assistant", "content": assistant_prompt})

        function_to_run = None
        package = None
        function_to_run = self.get_chat_completion
        package = messages

        try:
            self.completion = func_timeout(
                self.timeout,
                function_to_run,
                args=(self.model, package, 400, 0),
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
