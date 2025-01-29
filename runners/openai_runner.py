from time import time
import json
import os
import pandas as pd

from runners.base_runner import generate_base_prompt, extract_sql_from_response, run_eval_in_threadpool
from utils.questions import prepare_questions_df
from utils.llm import chat_openai
from utils.creds import db_creds_all
from utils.reporting import upload_results
from eval.eval import compare_query_results


def generate_prompt(
    prompt_file,
    question,
    db_name,
    db_type,
    instructions="",
    k_shot_prompt="",
    glossary="",
    table_metadata_string="",
    prev_invalid_sql="",
    prev_error_msg="",
    public_data=True,
    shuffle=True,
):
    """OpenAI-specific prompt handling"""
    # Get base prompt data
    base_data = generate_base_prompt(
        prompt_file, question, db_name, db_type, instructions,
        k_shot_prompt, glossary, table_metadata_string,
        prev_invalid_sql, prev_error_msg, public_data, shuffle
    )

    # Load and format OpenAI-specific JSON prompt
    with open(prompt_file, "r") as f:
        prompt = json.load(f)

    pruned_metadata_str = base_data["table_metadata_string"]
    
    if prompt[0]["role"] == "system":
        prompt[0]["content"] = prompt[0]["content"].format(
            db_type=db_type,
        )
        prompt[1]["content"] = prompt[1]["content"].format(
            user_question=question,
            instructions=instructions,
            table_metadata_string=pruned_metadata_str,
            k_shot_prompt=k_shot_prompt,
        )
    else:
        prompt[0]["content"] = prompt[0]["content"].format(
            db_type=db_type,
            user_question=question,
            instructions=instructions,
            table_metadata_string=pruned_metadata_str,
            k_shot_prompt=k_shot_prompt,
        )
    return prompt


def process_row(row, model_name, args):
    """Process a single row using OpenAI"""
    start_time = time()
    messages = generate_prompt(
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
        shuffle=args.shuffle_metadata,
    )
    try:
        response = chat_openai(messages=messages, model=model_name, temperature=0.0)
        generated_query = extract_sql_from_response(response.content)
        
        result = {
            "generated_query": generated_query,
            "reason": "",
            "error_msg": "",
            "latency_seconds": time() - start_time,
            "tokens_used": response.input_tokens + response.output_tokens,
        }
        
        # Verify results
        expected_query = row["query"]
        db_name = row["db_name"]
        db_type = row["db_type"]
        try:
            is_correct = compare_query_results(
                query_gold=expected_query,
                query_gen=generated_query,
                db_name=db_name,
                db_type=db_type,
                db_creds=db_creds_all[db_type],
                question=row["question"],
                query_category=row["query_category"],
                decimal_points=args.decimal_points if hasattr(args, 'decimal_points') else 2,
            )
            if is_correct:
                row["is_correct"] = 1
            else:
                row["is_correct"] = 0
                result["error_msg"] = "INCORRECT RESULTS"
        except Exception as e:
            row["error_db_exec"] = 1
            result["error_msg"] = f"EXECUTION ERROR: {str(e)}"
            
        # Update row with result data
        row.update(result)
        return row
    except Exception as e:
        row["error_query_gen"] = 1
        row["generated_query"] = ""
        row["reason"] = ""
        row["error_msg"] = f"GENERATION ERROR: {str(e)}"
        row["latency_seconds"] = time() - start_time
        row["tokens_used"] = 0
        return row


def run_openai_eval(args):
    """Run evaluation using OpenAI"""
    questions_file_list = args.questions_file
    prompt_file_list = args.prompt_file
    output_file_list = args.output_file
    num_questions = args.num_questions
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
                mean_correct=("is_correct", "mean"),
                mean_error_db_exec=("error_db_exec", "mean"),
            )
            .reset_index()
        )
        print(agg_stats)

        # Create output directory if needed
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        output_df.to_csv(output_file, index=False, float_format="%.2f")

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
                    runner_type="openai",
                    prompt=prompt,
                    args=args,
                )
        except Exception as e:
            print(f"Error uploading results: {e}")