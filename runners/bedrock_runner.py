import json
import os
import pandas as pd
from tqdm import tqdm
from time import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3

from runners.base_runner import BaseRunner
from utils.gen_prompt import generate_prompt
from utils.questions import prepare_questions_df
from utils.creds import db_creds_all
from eval.eval import compare_query_results
from utils.reporting import upload_results


class BedrockRunner(BaseRunner):
    def __init__(self):
        super().__init__()
        self.client = boto3.client(service_name="bedrock-runtime")

    def _call_llm(self, prompt, model_name, temperature=0.0):
        """Call AWS Bedrock API."""
        body = json.dumps(
            {
                "prompt": prompt,
                "max_gen_len": 600,
                "temperature": 0,
                "top_p": 1,
            }
        )

        accept = "application/json"
        contentType = "application/json"
        response = self.client.invoke_model(
            body=body, modelId=model_name, accept=accept, contentType=contentType
        )
        model_response = json.loads(response["body"].read())

        # Create a response object similar to other runners
        class BedrockResponse:
            def __init__(self, content):
                self.content = content
                self.input_tokens = 0  # Bedrock doesn't provide token counts this way
                self.output_tokens = 0

        return BedrockResponse(model_response["generation"])

    def _extract_query(self, response_content):
        """Extract SQL query from response."""
        return (
            response_content.split("```sql")[-1].split("```")[0].split(";")[0].strip()
            + ";"
        )

    def process_row(self, row, model_name, args):
        """Override process_row to use simple handling."""
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
                question_0=row.get("question_0", ""),
                query_0=row.get("query_0", ""),
                question_1=row.get("question_1", ""),
                query_1=row.get("query_1", ""),
                cot_instructions=row.get("cot_instructions", ""),
                cot_pregen=row.get("cot_pregen", False),
                public_data=not args.use_private_data,
                columns_to_keep=args.num_columns,
                shuffle_metadata=args.shuffle_metadata,
            )

            response = self._call_llm(prompt, model_name)
            generated_query = self._extract_query(response.content)

            result = {
                "generated_query": generated_query,
                "latency_seconds": time() - start_time,
                "tokens_used": None,  # Bedrock doesn't provide token counts
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
                "tokens_used": None,
            }

    def run_eval(self, args):
        """Bedrock-specific evaluation logic."""
        for questions_file, prompt_file, output_file in zip(
            args.questions_file, args.prompt_file, args.output_file
        ):
            print(f"Using prompt file {prompt_file}")
            print("Preparing questions...")
            print(
                f"Using {'all' if args.num_questions is None else args.num_questions} question(s) from {questions_file}"
            )
            df = prepare_questions_df(
                questions_file,
                args.db_type,
                args.num_questions,
                args.k_shot,
                args.cot_table_alias,
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

            print(
                output_df.groupby("query_category")[["correct", "error_db_exec"]].mean()
            )
            output_df = output_df.sort_values(
                by=["db_name", "query_category", "question"]
            )

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
                    runner_type="bedrock_runner",
                    prompt=prompt,
                    args=args,
                )


def run_bedrock_eval(args):
    runner = BedrockRunner()
    runner.run_eval(args)
