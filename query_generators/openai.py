import datetime
import json
import os
import time
from typing import Dict, List
from func_timeout import FunctionTimedOut, func_timeout
import openai
import tiktoken

import yaml
from query_generators.query_generator import QueryGenerator
from utils.pruning import prune_metadata_str


class OpenAIChatQueryGenerator(QueryGenerator):
    """
    Query generator that uses OpenAI's chat models
    Models available: gpt-3.5-turbo-0613, gpt-4-0613
    """

    def __init__(
        self,
        db_creds: Dict[str, str],
        model: str,
        prompt_file: str,
        timeout: int,
        verbose: bool,
        **kwargs,
    ):
        self.db_creds = db_creds
        self.db_name = db_creds["database"]
        self.model = model
        self.prompt_file = prompt_file

        self.timeout = timeout
        self.verbose = verbose

    def get_chat_completion(
        self,
        model,
        messages,
        max_tokens=600,
        temperature=0,
        top_p=0.1,
        stop=["```"],
        logit_bias={},
    ):
        """Get OpenAI chat completion for a given prompt and model"""
        generated_text = ""
        try:
            completion = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                logit_bias=logit_bias,
            )
            generated_text = completion["choices"][0]["message"]["content"]
        except (openai.error.RateLimitError, openai.error.ServiceUnavailableError) as e:
            if self.verbose:
                print("Model overloaded. Pausing for 5s before retrying...")
            time.sleep(5)
            # Retry the api call after 5s
            completion = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                logit_bias=logit_bias,
            )
            generated_text = completion["choices"][0]["message"]["content"]
        except Exception as e:
            if self.verbose:
                print(type(e), e)
        return generated_text

    @staticmethod
    def count_tokens(model: str, messages: List[Dict[str, str]]) -> int:
        tokenizer = tiktoken.encoding_for_model(model)
        num_tokens = 0
        for message in messages:
            for _, value in message.items():
                num_tokens += len(tokenizer.encode(value))
        return num_tokens

    def generate_query(self, question: str) -> dict:
        start_time = time.time()
        self.err = ""
        self.query = ""
        self.reason = ""

        with open(self.prompt_file) as file:
            chat_prompt = file.read()

        sys_prompt = chat_prompt.split("### Input:")[0]
        user_prompt = chat_prompt.split("### Input:")[1].split("### Generated SQL:")[0]
        assistant_prompt = chat_prompt.split("### Generated SQL:")[1]

        user_prompt = user_prompt.format(
            user_question=question,
            table_metadata_string=prune_metadata_str(question, self.db_name),
        )

        messages = []
        messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": user_prompt})
        messages.append({"role": "assistant", "content": assistant_prompt})

        try:
            self.completion = func_timeout(
                self.timeout,
                self.get_chat_completion,
                args=(
                    self.model,
                    messages,
                    400,
                    0,
                    0.1,
                    ["```"],
                ),
            )
            # results = yaml.safe_load(self.completion)
            # self.query = results["sql"]
            # self.reason = results["reason_for_query"]
            results = self.completion
            self.query = results.split("```sql")[-1]
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
            "tokens_used": self.count_tokens(self.model, messages),
        }
