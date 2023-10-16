from eval.eval import compare_query_results
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
    pipeline,
)
from utils.pruning import prune_metadata_str
from utils.questions import prepare_questions_df
from tqdm import tqdm
from psycopg2.extensions import QueryCanceledError
from time import time
import gc
from peft import PeftModel, PeftConfig


def generate_prompt(prompt_file, question, db_name, public_data):
    with open(prompt_file, "r") as f:
        prompt = f.read()

    pruned_metadata_str = prune_metadata_str(question, db_name, public_data)
    prompt = prompt.format(
        user_question=question, table_metadata_string=pruned_metadata_str
    )
    return prompt


def get_tokenizer_model(model_name):
    if "llama" not in model_name:
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        # model = AutoModelForCausalLM.from_pretrained(
        #     model_name,
        #     trust_remote_code=True,
        #     torch_dtype=torch.float16,
        #     device_map="auto",
        #     use_cache=True
        # )
        model_path = "/home/defog/finetuning/starcoder/sqlcoder_npl_cfc_map_600"
        config = PeftConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            use_auth_token=True,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
    else:
        tokenizer = LlamaTokenizer.from_pretrained(
            model_name, legacy=False, use_fast=True
        )
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            use_cache=True,
            use_flash_attention_2=True,
        )
    return tokenizer, model


def run_hf_eval(
    questions_file: str,
    prompt_file: str,
    num_questions: int = None,
    public_data: bool = True,
    model_name: str = "defog/starcoder-finetune-v3",
    output_file: str = "results.csv",
):
    print("preparing questions...")
    # get questions
    df = prepare_questions_df(questions_file, num_questions)

    # create a prompt for each question
    df["prompt"] = df[["question", "db_name"]].apply(
        lambda row: generate_prompt(
            prompt_file, row["question"], row["db_name"], public_data
        ),
        axis=1,
    )

    print("questions prepared\nnow loading model...")
    # initialize tokenizer and model
    tokenizer, model = get_tokenizer_model(model_name)
    model.tie_weights()

    print("model loaded\nnow generating and evaluating predictions...")

    # from here, we generate and evaluate predictions
    # eos_token_id = tokenizer.convert_tokens_to_ids(["```"])[0]
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    total_tried = 0
    total_correct = 0
    output_rows = []

    if "llama" not in model_name.lower():
        pipeline_config = {
            "max_new_tokens": 300,
            "do_sample": False,
            "num_beams": 5,
        }
    else:
        pipeline_config = {
            "max_new_tokens": 300,
            "do_sample": False,
            "num_beams": 3,
        }

    with tqdm(total=len(df)) as pbar:
        for row in df.to_dict("records"):
            total_tried += 1
            start_time = time()
            generated_query = (
                pipe(
                    row["prompt"],
                    num_return_sequences=1,
                    # eos_token_id=eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                    **pipeline_config,
                )[0]["generated_text"]
                .split("```sql")[-1]
                .split("```")[0]
                .split(";")[0]
                .strip()
                + ";"
            )
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            end_time = time()

            row["generated_query"] = generated_query
            row["latency_seconds"] = end_time - start_time
            golden_query = row["query"]
            db_name = row["db_name"]
            question = row["question"]
            query_category = row["query_category"]
            exact_match = correct = 0
            db_creds = {
                "host": "localhost",
                "port": 5432,
                "user": "postgres",
                "password": "postgres",
                "database": db_name,
            }

            try:
                exact_match, correct = compare_query_results(
                    query_gold=golden_query,
                    query_gen=generated_query,
                    db_name=db_name,
                    db_creds=db_creds,
                    question=question,
                    query_category=query_category,
                )
                row["exact_match"] = int(exact_match)
                row["correct"] = int(correct)
                row["error_msg"] = ""
                if correct:
                    total_correct += 1
            except QueryCanceledError as e:
                row["timeout"] = 1
                row["error_msg"] = f"QUERY EXECUTION TIMEOUT: {e}"
            except Exception as e:
                row["error_db_exec"] = 1
                row["error_msg"] = f"QUERY EXECUTION ERROR: {e}"

            output_rows.append(row)
            pbar.update(1)
            pbar.set_description(
                f"Correct so far: {total_correct}/{total_tried} ({100*total_correct/total_tried:.2f}%)"
            )

    output_df = pd.DataFrame(output_rows)
    del output_df["prompt"]
    print(output_df.groupby("query_category")[["exact_match", "correct"]].mean())
    output_df = output_df.sort_values(by=["db_name", "query_category", "question"])
    output_df.to_csv(output_file, index=False, float_format="%.2f")
