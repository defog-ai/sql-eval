import json
import os
import boto3
from time import time
import pandas as pd

from runners.base_runner import generate_base_prompt, extract_sql_from_response, run_eval_in_threadpool
from utils.gen_prompt import generate_prompt
from utils.questions import prepare_questions_df
from utils.creds import db_creds_all
from utils.reporting import upload_results
from eval.eval import compare_query_results

bedrock = boto3.client(service_name="bedrock-runtime")

def process_row(row, model_id, args):
    """Process a single row using AWS Bedrock"""
    start_time = time()
    try:
        # Bedrock-specific request payload
        body = json.dumps({
            "prompt": row["prompt"],
            "max_gen_len": 600,
            "temperature": 0,
            "top_p": 1,
        })

        accept = "application/json"
        contentType = "application/json"
        response = bedrock.invoke_model(
            body=body, modelId=model_id, accept=accept, contentType=contentType
        )
        model_response = json.loads(response["body"].read())
        generated_query = model_response["generation"]
        end_time = time()

        # Bedrock-specific SQL extraction
        generated_query = extract_sql_from_response(generated_query)

        # Store results
        row["generated_query"] = generated_query
        row["latency_seconds"] = end_time - start_time
        row["tokens_used"] = None  # Bedrock doesn't provide token count
        
        # Verify results
        golden_query = row["query"]
        db_name = row["db_name"]
        db_type = row["db_type"]
        question = row["question"]
        query_category = row["query_category"]
        table_metadata_string = row["table_metadata_string"]
        
        try:
            exact_match, correct = compare_query_results(
                query_gold=golden_query,
                query_gen=generated_query,
                db_name=db_name,
                db_type=db_type,
                db_creds=db_creds_all[db_type],
                question=question,
                query_category=query_category,
                table_metadata_string=table_metadata_string,
                decimal_points=args.decimal_points if hasattr(args, 'decimal_points') else 2,
            )
            row["exact_match"] = int(exact_match)
            row["correct"] = int(correct)
            row["is_correct"] = int(correct)  # For compatibility with base runner
            row["error_msg"] = ""
        except Exception as e:
            row["error_db_exec"] = 1
            row["error_msg"] = f"QUERY EXECUTION ERROR: {e}"
            
        return row
    except Exception as e:
        row["error_query_gen"] = 1
        row["generated_query"] = ""
        row["error_msg"] = f"GENERATION ERROR: {e}"
        row["latency_seconds"] = time() - start_time
        row["tokens_used"] = None
        return row


def run_bedrock_eval(args):
    """Run evaluation using AWS Bedrock"""
    questions_file_list = args.questions_file
    prompt_file_list = args.prompt_file
    output_file_list = args.output_file
    num_questions = args.num_questions
    public_data = not args.use_private_data
    k_shot = args.k_shot
    db_type = args.db_type
    cot_table_alias = args.cot_table_alias

    for questions_file, prompt_file, output_file in zip(
        questions_file_list, prompt_file_list, output_file_list
    ):
        print(f"Using prompt file {prompt_file}")
        print("Preparing questions...")
        print(
            f"Using {'all' if num_questions is None else num_questions} question(s) from {questions_file}"
        )
        df = prepare_questions_df(
            questions_file, db_type, num_questions, k_shot, cot_table_alias
        )

        # Create prompts with all parameters
        df["prompt"] = df.apply(
            lambda row: generate_prompt(
                prompt_file,
                row["question"],
                row["db_name"],
                row["db_type"],
                row["instructions"],
                row["k_shot_prompt"],
                row["glossary"],
                row["table_metadata_string"],
                row["prev_invalid_sql"],
                row["prev_error_msg"],
                row.get("question_0", ""),
                row.get("query_0", ""),
                row.get("question_1", ""),
                row.get("query_1", ""),
                row.get("cot_instructions", ""),
                row.get("cot_pregen", False),
                public_data,
                args.num_columns if hasattr(args, 'num_columns') else 40,
                args.shuffle_metadata,
                row.get("table_aliases", ""),
            ),
            axis=1,
        )
        
        output_rows, total_correct, total_tried = run_eval_in_threadpool(
            df, args.model, process_row, args
        )

        # Convert to DataFrame and save results
        output_df = pd.DataFrame(output_rows)
        output_df = output_df.sort_values(by=["db_name", "query_category", "question"])
        if "prompt" in output_df.columns:
            del output_df["prompt"]

        # Get stats by query category
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

        # Create output directory if needed
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        try:
            output_df.to_csv(output_file, index=False, float_format="%.2f")
        except:
            output_df.to_pickle(output_file)

        # Print summary stats
        print(f"Total questions: {total_tried}")
        print(f"Total correct: {total_correct}")
        print(f"Accuracy: {total_correct/total_tried:.3f}")

        # Upload results if URL provided
        try:
            if hasattr(args, 'upload_url') and args.upload_url:
                with open(prompt_file, "r") as f:
                    prompt = f.read()
                upload_results(
                    results=output_df.to_dict("records"),
                    url=args.upload_url,
                    runner_type="bedrock",
                    prompt=prompt,
                    args=args,
                )
        except Exception as e:
            print(f"Error uploading results: {e}")