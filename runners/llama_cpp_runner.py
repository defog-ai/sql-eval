import os
from time import time
import pandas as pd
from llama_cpp import Llama
from tqdm import tqdm

from utils.gen_prompt import generate_prompt
from utils.questions import prepare_questions_df
from utils.creds import db_creds_all
from utils.reporting import upload_results
from eval.eval import compare_query_results


def process_row(llm, row, args):
    """Process a single row using Llama.cpp"""
    start_time = time()
    try:
        prompt = row["prompt"]
        response = llm(
            prompt,
            max_tokens=512,
            temperature=0,
            top_p=1,
            echo=False,
            repeat_penalty=1.0,
        )
        # Llama.cpp-specific SQL extraction
        generated_query = response["choices"][0]["text"].split(";")[0].split("```")[0].strip() + ";"
        end_time = time()

        # Store results
        row["generated_query"] = generated_query
        row["latency_seconds"] = end_time - start_time
        
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
        return row


def run_llama_cpp_eval(args):
    """Run evaluation using Llama.cpp"""
    questions_file_list = args.questions_file
    prompt_file_list = args.prompt_file
    output_file_list = args.output_file
    num_questions = args.num_questions
    public_data = not args.use_private_data
    model_path = args.model
    k_shot = args.k_shot
    db_type = args.db_type
    cot_table_alias = args.cot_table_alias

    # Load Llama.cpp model
    llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=4096)

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

        # Process rows with direct iteration (no threading)
        total_tried = 0
        total_correct = 0
        output_rows = []

        with tqdm(total=len(df)) as pbar:
            for row in df.to_dict("records"):
                row = process_row(llm, row, args)
                output_rows.append(row)
                if row.get("correct", 0):
                    total_correct += 1
                total_tried += 1
                pbar.update(1)
                pbar.set_description(
                    f"Correct so far: {total_correct}/{total_tried} ({100*total_correct/total_tried:.2f}%)"
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
                    runner_type="llama_cpp_runner",
                    prompt=prompt,
                    args=args,
                )
        except Exception as e:
            print(f"Error uploading results: {e}")