import os
import pandas as pd
from tqdm import tqdm
from time import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

from runners.base_runner import BaseRunner
from utils.gen_prompt import generate_prompt as base_generate_prompt
from utils.questions import prepare_questions_df
from utils.creds import db_creds_all
from eval.eval import compare_query_results
from utils.reporting import upload_results


class MistralRunner(BaseRunner):
    def __init__(self):
        super().__init__()
        self.api_key = os.environ.get("MISTRAL_API_KEY")
        self.client = MistralClient(api_key=self.api_key)

    def generate_prompt(self, prompt_file, **kwargs):
        """Mistral-specific prompt generation."""
        with open(prompt_file, "r") as f:
            prompt = f.read()

        # Check that System and User prompts are in the prompt file
        if "System:" not in prompt or "User:" not in prompt:
            raise ValueError("Invalid prompt file. Please use prompt_mistral.md")
        
        sys_prompt = prompt.split("System:")[1].split("User:")[0].strip()
        user_prompt = prompt.split("User:")[1].strip()

        # Get table metadata using base generate_prompt
        table_metadata_str = base_generate_prompt(
            prompt_file=prompt_file,
            question=kwargs.get("question", ""),
            db_name=kwargs.get("db_name", ""),
            db_type=kwargs.get("db_type", ""),
            instructions=kwargs.get("instructions", ""),
            k_shot_prompt=kwargs.get("k_shot_prompt", ""),
            glossary=kwargs.get("glossary", ""),
            table_metadata_string=kwargs.get("table_metadata_string", ""),
            prev_invalid_sql=kwargs.get("prev_invalid_sql", ""),
            prev_error_msg=kwargs.get("prev_error_msg", ""),
            public_data=kwargs.get("public_data", True),
            shuffle_metadata=kwargs.get("shuffle_metadata", True),
        )

        # Format user prompt
        user_prompt = user_prompt.format(
            user_question=kwargs.get("question", ""),
            instructions=kwargs.get("instructions", ""),
            table_metadata_string=table_metadata_str,
            k_shot_prompt=kwargs.get("k_shot_prompt", ""),
            glossary=kwargs.get("glossary", ""),
            prev_invalid_sql=kwargs.get("prev_invalid_sql", ""),
            prev_error_msg=kwargs.get("prev_error_msg", ""),
        )

        return [
            ChatMessage(role="system", content=sys_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]

    def _call_llm(self, messages, model_name, temperature=0.0):
        """Call Mistral API."""
        chat_response = self.client.chat(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=600,
        )
        # Create a response object similar to other runners
        class MistralResponse:
            def __init__(self, content):
                self.content = content
                self.input_tokens = 0  # Mistral doesn't provide token counts in this way
                self.output_tokens = 0

        return MistralResponse(chat_response.choices[0].message.content)

    def _extract_query(self, response_content):
        """Extract SQL query from response with Mistral-specific handling."""
        try:
            # Replace backslashes 
            content = response_content.replace("\\", "")
            # First try to extract from SQL code blocks
            query = content.split(";")[0].split("```sql")[-1].strip()
            query = [i for i in query.split("```") if i.strip() != ""][0] + ";"
            return query
        except Exception as e:
            # Fallback to raw content
            print(f"Query extraction error: {e}")
            return response_content.split(";")[0].strip() + ";"

    def process_row(self, row, model_name, args):
        """Override process_row to use Mistral prompt generation."""
        start_time = time()
        try:
            messages = self.generate_prompt(
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

            response = self._call_llm(messages, model_name)
            generated_query = self._extract_query(response.content)

            result = {
                "generated_query": generated_query,
                "latency_seconds": time() - start_time,
                "tokens_used": response.input_tokens + response.output_tokens
            }

            # Run comparison
            try:
                exact_match, correct = compare_query_results(
                    query_gold=row["query"],
                    query_gen=generated_query,
                    db_name=row["db_name"],
                    db_type=row["db_type"],
                    db_creds=db_creds_all[row["db_type"]],
                    question=row["question"],
                    query_category=row["query_category"],
                    table_metadata_string=row["table_metadata_string"],
                    decimal_points=args.decimal_points,
                )
                result["exact_match"] = int(exact_match)
                result["correct"] = int(correct)
                result["error_msg"] = ""
            except Exception as e:
                result["error_db_exec"] = 1
                result["error_msg"] = f"QUERY EXECUTION ERROR: {str(e)}"

            return {**row, **result}

        except Exception as e:
            return {
                **row,
                "generated_query": "",
                "exact_match": 0,
                "correct": 0,
                "error_db_exec": 1,
                "error_msg": f"PROCESSING ERROR: {str(e)}",
                "latency_seconds": time() - start_time,
                "tokens_used": 0
            }

    def run_eval(self, args):
        """Mistral-specific evaluation logic."""
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY environment variable must be set")

        for questions_file, prompt_file, output_file in zip(
            args.questions_file, args.prompt_file, args.output_file
        ):
            print(f"Using prompt file {prompt_file}")
            print("Preparing questions...")
            print(
                f"Using {'all' if args.num_questions is None else args.num_questions} question(s) from {questions_file}"
            )
            df = prepare_questions_df(
                questions_file, args.db_type, args.num_questions, args.k_shot, args.cot_table_alias
            )

            # Process all rows
            output_rows = []
            total_tried = total_correct = 0
            
            with ThreadPoolExecutor(args.parallel_threads) as executor:
                futures = []
                for row in df.to_dict("records"):
                    futures.append(
                        executor.submit(self.process_row, row, args.model, args)
                    )

                with tqdm(as_completed(futures), total=len(futures)) as pbar:
                    for f in pbar:
                        row = f.result()
                        output_rows.append(row)
                        if row.get("correct", 0):
                            total_correct += 1
                        total_tried += 1
                        pbar.set_description(
                            f"Correct so far: {total_correct}/{total_tried} ({100*total_correct/total_tried:.2f}%)"
                        )

            # Save results
            output_df = pd.DataFrame(output_rows)
            if "prompt" in output_df.columns:
                del output_df["prompt"]
                
            print(output_df.groupby("query_category")[["correct", "error_db_exec"]].mean())
            output_df = output_df.sort_values(by=["db_name", "query_category", "question"])
            
            # Save to file
            output_dir = os.path.dirname(output_file)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            try:
                output_df.to_csv(output_file, index=False, float_format="%.2f")
            except:
                output_df.to_pickle(output_file)
            print(f"Saved results to {output_file}")

            # Upload results if needed
            if args.upload_url is not None:
                with open(prompt_file, "r") as f:
                    prompt = f.read()
                upload_results(
                    results=output_df.to_dict("records"),
                    url=args.upload_url,
                    runner_type="mistral_runner",
                    prompt=prompt,
                    args=args,
                )


def run_mistral_eval(args):
    runner = MistralRunner()
    runner.run_eval(args)