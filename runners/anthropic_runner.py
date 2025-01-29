import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.llm import chat_anthropic
from utils.questions import prepare_questions_df
from utils.reporting import upload_results
from utils.creds import db_creds_all
from eval.eval import compare_query_results
from runners.base_runner import BaseRunner, generate_prompt


class AnthropicRunner(BaseRunner):
    def _call_llm(self, prompt, model_name, temperature=0.0):
        """Call Anthropic API, handling both JSON and text prompts."""
        if isinstance(prompt, list):  # JSON prompt format
            messages = prompt
        else:  # Text prompt format
            messages = [{"role": "user", "content": prompt}]
        return chat_anthropic(messages=messages, model=model_name, temperature=temperature)

    def _extract_query(self, response_content):
        """Extract SQL query from response."""
        try:
            return response_content.split("```sql", 1)[-1].split("```", 1)[0].strip()
        except:
            # Fallback to extract anything that looks like SQL
            return response_content.split(";")[0].strip() + ";"

    def run_eval(self, args):
        """Anthropic-specific evaluation logic."""
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
            
            with ThreadPoolExecutor(args.parallel_threads) as executor:
                futures = []
                for row in input_rows:
                    generated_query_fut = executor.submit(
                        self.process_row,
                        row=row,
                        model_name=args.model,
                        args=args,
                    )
                    futures.append(generated_query_fut)

                total_tried = 0
                total_correct = 0
                for f in (pbar := tqdm(as_completed(futures), total=len(futures))):
                    total_tried += 1
                    i = futures.index(f)
                    row = input_rows[i]
                    result_dict = f.result()
                    query_gen = result_dict["query"]
                    reason = result_dict["reason"]
                    err = result_dict["err"]
                    # save custom metrics
                    if "latency_seconds" in result_dict:
                        row["latency_seconds"] = result_dict["latency_seconds"]
                    if "tokens_used" in result_dict:
                        row["tokens_used"] = result_dict["tokens_used"]
                    row["generated_query"] = query_gen
                    row["reason"] = reason
                    row["error_msg"] = err
                    # save failures into relevant columns in the dataframe
                    if "GENERATION ERROR" in err:
                        row["error_query_gen"] = 1
                    else:
                        expected_query = row["query"]
                        db_name = row["db_name"]
                        db_type = row["db_type"]
                        try:
                            is_correct = compare_query_results(
                                query_gold=expected_query,
                                query_gen=query_gen,
                                db_name=db_name,
                                db_type=db_type,
                                question=row["question"],
                                query_category=row["query_category"],
                                db_creds=db_creds_all[db_type],
                            )
                            if is_correct:
                                total_correct += 1
                                row["is_correct"] = 1
                                row["error_msg"] = ""
                            else:
                                row["is_correct"] = 0
                                row["error_msg"] = "INCORRECT RESULTS"
                        except Exception as e:
                            row["error_db_exec"] = 1
                            row["error_msg"] = f"EXECUTION ERROR: {str(e)}"
                    output_rows.append(row)
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
                    mean_correct=("is_correct", "mean"),
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
                    runner_type="anthropic",
                    prompt=prompt,
                    args=args,
                )

def run_anthropic_eval(args):
    runner = AnthropicRunner()
    runner.run_eval(args)