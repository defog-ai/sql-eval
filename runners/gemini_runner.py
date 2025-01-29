import os
import pandas as pd
from tqdm import tqdm
from time import time
import sqlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.llm import chat_gemini
from utils.questions import prepare_questions_df
from utils.reporting import upload_results
from utils.creds import db_creds_all
from eval.eval import compare_query_results
from runners.base_runner import BaseRunner, generate_prompt


class GeminiRunner(BaseRunner):
    def _call_llm(self, prompt, model_name, temperature=0.0):
        """Call Gemini API, handling both JSON and text prompts."""
        if isinstance(prompt, list):  # JSON prompt format
            messages = prompt
        else:  # Text prompt format
            messages = [{"role": "user", "content": prompt}]
        return chat_gemini(messages=messages, model=model_name, temperature=temperature)

    def _extract_query(self, response_content):
        """Extract SQL query from response."""
        try:
            return response_content.split("```sql", 1)[-1].split("```", 1)[0].strip()
        except:
            # Fallback to extract anything that looks like SQL
            try:
                return sqlparse.format(
                    response_content.split(";")[0].strip() + ";",
                    strip_comments=True,
                    strip_whitespace=True,
                    keyword_case="upper",
                )
            except:
                return response_content.split(";")[0].strip() + ";"

    def process_row(self, row, model_name, args):
        """Gemini-specific row processing logic."""
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
                public_data=not args.use_private_data,
                shuffle_metadata=args.shuffle_metadata,
            )

            response = self._call_llm(prompt, model_name)
            generated_query = self._extract_query(response.content)
            
            row["generated_query"] = generated_query
            row["latency_seconds"] = time() - start_time
            row["tokens_used"] = response.input_tokens + response.output_tokens

            # Run comparison
            golden_query = row["query"]
            db_name = row["db_name"]
            db_type = row["db_type"]
            question = row["question"]
            query_category = row["query_category"]

            try:
                exact_match, correct = compare_query_results(
                    query_gold=golden_query,
                    query_gen=generated_query,
                    db_name=db_name,
                    db_type=db_type,
                    db_creds=db_creds_all[db_type],
                    question=question,
                    query_category=query_category,
                    decimal_points=args.decimal_points,
                )
                row["exact_match"] = int(exact_match)
                row["correct"] = int(correct)
                row["error_msg"] = ""
            except Exception as e:
                row["error_db_exec"] = 1
                row["error_msg"] = f"QUERY EXECUTION ERROR: {e}"

            return row

        except Exception as e:
            row["error_db_exec"] = 1
            row["error_msg"] = f"GENERATION ERROR: {e}"
            row["latency_seconds"] = time() - start_time
            row["tokens_used"] = 0
            return row

    def run_eval(self, args):
        """Gemini-specific evaluation logic."""
        questions_file_list = args.questions_file
        prompt_file_list = args.prompt_file
        output_file_list = args.output_file
        
        for questions_file, prompt_file, output_file in zip(
            questions_file_list, prompt_file_list, output_file_list
        ):
            print(f"Using prompt file {prompt_file}")
            print("Preparing questions...")
            print(
                f"Using {'all' if args.num_questions is None else args.num_questions} question(s) from {questions_file}"
            )
            question_query_df = prepare_questions_df(
                questions_file, args.db_type, args.num_questions, args.k_shot, args.cot_table_alias
            )
            input_rows = question_query_df.to_dict("records")
            output_rows = []
            
            total_tried = 0
            total_correct = 0
            
            with ThreadPoolExecutor(args.parallel_threads) as executor:
                futures = []
                for row in input_rows:
                    futures.append(executor.submit(
                        self.process_row,
                        row=row,
                        model_name=args.model,
                        args=args
                    ))

                with tqdm(as_completed(futures), total=len(futures)) as pbar:
                    for f in pbar:
                        row = f.result()
                        output_rows.append(row)
                        if row.get("correct", 0):
                            total_correct += 1
                        total_tried += 1
                        pbar.set_description(
                            f"Accuracy: {round(total_correct/total_tried * 100, 2)}% ({total_correct}/{total_tried})"
                        )

            # save results to csv
            output_df = pd.DataFrame(output_rows)
            output_df = output_df.sort_values(by=["db_name", "query_category", "question"])
            if "prompt" in output_df.columns:
                del output_df["prompt"]
                
            # get num rows, mean correct, mean error_db_exec for each query_category
            agg_stats = (
                output_df.groupby("query_category")
                .agg(
                    num_rows=("db_name", "count"),
                    mean_correct=("correct", "mean"),
                    mean_error_db_exec=("error_db_exec", "mean"),
                )
                .reset_index()
            )
            print(agg_stats)
            
            # get directory of output_file and create if not exist
            output_dir = os.path.dirname(output_file)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_df.to_csv(output_file, index=False, float_format="%.2f")

            # get average rate of correct results
            avg_subset = output_df["correct"].sum() / len(output_df)
            print(f"Average correct rate: {avg_subset:.2f}")

            results = output_df.to_dict("records")

            # upload results
            with open(prompt_file, "r") as f:
                prompt = f.read()
            if args.upload_url is not None:
                upload_results(
                    results=results,
                    url=args.upload_url,
                    runner_type="gemini",
                    prompt=prompt,
                    args=args,
                )

def run_gemini_eval(args):
    runner = GeminiRunner()
    runner.run_eval(args)