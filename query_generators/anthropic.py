import time
from typing import Dict, List
from func_timeout import FunctionTimedOut, func_timeout
from anthropic import Anthropic
import os

from query_generators.query_generator import QueryGenerator
from utils.pruning import prune_metadata_str

anthropic = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


class AnthropicQueryGenerator(QueryGenerator):
    """
    Query generator that uses Anthropic's models
    Models available: claude-2, claude-instant-1
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
        try:
            completion = anthropic.completions.create(
                model=model,
                prompt=prompt,
                max_tokens_to_sample=max_tokens,
                temperature=temperature,
                stop_sequences=stop,
            )
            generated_text = completion.completion
        except Exception as e:
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
        self, question: str, instructions: str, k_shot_prompt: str, glossary: str, table_metadata_string: str
    ) -> dict:
        start_time = time.time()
        self.err = ""
        self.query = ""
        self.reason = ""

        with open(self.prompt_file) as file:
            model_prompt = file.read()
            # Check that Human and Assistant prompts are in the prompt file
            if "Human:" not in model_prompt:
                raise ValueError("Invalid prompt file. Please use prompt_anthropic.md")
        question_instructions = question + " " + instructions
        if table_metadata_string == "":
            pruned_metadata_str = prune_metadata_str(
                question_instructions, self.db_name, self.use_public_data
            )
        else:
            pruned_metadata_str = table_metadata_string
        prompt = model_prompt.format(
            user_question=question,
            table_metadata_string=pruned_metadata_str,
            instructions=instructions,
            k_shot_prompt=k_shot_prompt,
            glossary=glossary,
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
                    400,
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
