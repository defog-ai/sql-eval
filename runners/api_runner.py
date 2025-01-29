import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd
import sqlparse
import re
import requests
from tqdm import tqdm
from time import time

from runners.base_runner import BaseRunner
from utils.gen_prompt import generate_prompt
from utils.questions import prepare_questions_df
from utils.creds import db_creds_all
from eval.eval import compare_query_results
from utils.reporting import upload_results


def clean_generated_query(query: str):
    """Clean up the generated query with post-processing heuristics."""
    query = sqlparse.format(query, reindent_aligned=True)
    query = query.replace("< =", "<=").replace("> =", ">=")
    query = query.replace("/ NULLIF (", "/ NULLIF (1.0 * ")
    query = re.sub(r"\s*\(\s*", "(", query)
    query = re.sub(r"\s*\)", ")", query)
    return query


class APIRunner(BaseRunner):
    def __init__(self):
        super().__init__()
        self.api_url = None
        self.api_type = None

    def _mk_vllm_json(self, prompt, num_beams, logprobs=False, sql_lora_path=None, sql_lora_name=None):
        payload = {
            "prompt": prompt,
            "n": 1,
            "use_beam_search": num_beams > 1,
            "best_of": num_beams,
            "temperature": 0,
            "stop": [";", "```"],
            "max_tokens": 4000,
            "seed": 42,
            "sql_lora_path": sql_lora_path,
            "sql_lora_name": sql_lora_name,
        }
        if logprobs:
            payload["logprobs"] = 2
        return payload

    def _mk_tgi_json(self, prompt, num_beams):
        return {
            "inputs": prompt,
            "parameters": {
                "best_of": num_beams,
                "do_sample": num_beams > 1,
                "return_full_text": False,
                "max_new_tokens": 1024,
            },
        }

    def _call_llm(self, prompt, model_name, temperature=0.0):
        """Call API endpoint."""
        # model_name is unused but kept for BaseRunner compatibility
        start_time = time()
        json_data = None
        
        if self.api_type == "tgi":
            json_data = self._mk_tgi_json(prompt, self.num_beams)
        elif self.api_type == "vllm":
            json_data = self._mk_vllm_json(
                prompt, self.num_beams, self.logprobs, self.sql_lora_path, self.sql_lora_name
            )
        else:
            json_data = {
                "prompt": prompt,
                "n": 1,
                "use_beam_search": self.num_beams > 1,
                "best_of": self.num_beams,
                "temperature": 0,
                "stop": [";", "```"],
                "max_tokens": 4000,
            }
            
        try:
            response = requests.post(
                self.api_url,
                json=json_data,
                timeout=200,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"API ERROR: {str(e)}")
        finally:
            self.request_time = time() - start_time

    def _extract_query(self, response_json, prompt):
        """Extract SQL query from API response."""
        if self.api_type == "tgi":
            try:
                return response_json["generated_text"]
            except KeyError:
                print(response_json)
                return ""
        elif "[SQL]" not in prompt:
            return (
                response_json["text"][0]
                .split("```sql")[-1]
                .split("```")[0]
                .split(";")[0]
                .strip()
                + ";"
            )
        else:
            generated_text = response_json["text"][0]
            if "[SQL]" in generated_text:
                generated_text = generated_text.split("[SQL]", 1)[1].strip()
            return generated_text.strip()

    def process_row(self, row, model_name, args):
        """API-specific row processing."""
        # Set API-specific attributes
        self.api_url = args.api_url
        self.api_type = args.api_type
        self.num_beams = args.num_beams
        self.logprobs = args.logprobs
        self.sql_lora_path = args.adapter
        self.sql_lora_name = args.adapter_name
        
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
                question_0=row.get("question_0", ""),
                query_0=row.get("query_0", ""),
                question_1=row.get("question_1", ""),
                query_1=row.get("query_1", ""),
                cot_instructions=row.get("cot_instructions", ""),
                cot_pregen=row.get("cot_pregen", False),
                public_data=not args.use_private_data,
                columns_to_keep=args.num_columns,
                shuffle_metadata=args.shuffle_metadata,
                table_aliases=row.get("table_aliases", ""),
            )

            response_json = self._call_llm(prompt, None)  # model_name unused for API
            generated_query = self._extract_query(response_json, prompt)
            generated_query = clean_generated_query(generated_query)

            # Handle logprobs if present
            logprobs_display = []
            if "logprobs" in response_json:
                for item in response_json["logprobs"]:
                    probs = list(item.values())
                    probs_to_append = {}
                    for prob in probs:
                        rank = prob["rank"]
                        logprob = prob["logprob"]
                        token = prob["decoded_token"]
                        probs_to_append.update({
                            f"rank_{rank}_token": token,
                            f"rank_{rank}_logprob": logprob,
                            f"rank_{rank}_prob": 10**logprob,
                        })
                    probs_to_append["prob_diff"] = (
                        probs_to_append["rank_1_prob"] - probs_to_append["rank_2_prob"]
                    )
                    logprobs_display.append(probs_to_append)

            # Prepare result
            result = {
                "generated_query": generated_query,
                "latency_seconds": self.request_time,
                "tokens_used": None,  # API doesn't provide token counts
                "logprobs": logprobs_display if self.logprobs else None
            }

            # Run comparison
            try:
                exact_match, correct = compare_query_results(
                    query_gold=row["query"],
                    query_gen=generated_query,
                    db_name=row["db_name"],
                    db_type=row["db_type"],
                    db_creds=db_creds_all.get(row["db_type"], {}),
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
                "error_msg": f"API ERROR: {str(e)}",
                "tokens_used": None,
                "latency_seconds": self.request_time if hasattr(self, 'request_time') else None,
                "logprobs": [] if self.logprobs else None
            }

    def run_eval(self, args):
        """API-specific evaluation logic."""
        # Validate API requirements
        if not args.api_url:
            raise ValueError("API URL must be provided for API runner")
        if not args.api_type or args.api_type not in ["vllm", "tgi"]:
            raise ValueError("API type must be one of 'vllm', 'tgi'")

        # Set up logprobs if needed
        if args.logprobs:
            if not os.path.exists("./eval-visualizer"):
                raise Exception(
                    "The eval-visualizer directory does not exist. Please clone it with "
                    "`git clone https://github.com/defog-ai/eval-visualizer/` before running "
                    "sql-eval with the --logprobs flag."
                )
            if not os.path.exists("./eval-visualizer/public"):
                os.makedirs("./eval-visualizer/public")

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
                        executor.submit(self.process_row, row, None, args)
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
            
            # Handle logprobs if needed
            if args.logprobs:
                print(
                    f"Writing logprobs to JSON file at eval-visualizer/public/{output_file.split('/')[-1].replace('.csv', '.json')}"
                )
                with open(
                    f"./eval-visualizer/public/{output_file.split('/')[-1].replace('.csv', '.json')}",
                    "w",
                ) as f:
                    json.dump(output_rows, f)

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
                run_name = args.run_name or output_file.split("/")[-1].replace(".csv", "")
                upload_results(
                    results=output_rows,
                    url=args.upload_url,
                    runner_type="api_runner",
                    args=args,
                    run_name=run_name,
                )


def run_api_eval(args):
    runner = APIRunner()
    runner.run_eval(args)