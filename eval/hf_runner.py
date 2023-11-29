import os
from typing import Optional
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


def generate_prompt(prompt_file, question, db_name):
    with open(prompt_file, "r") as f:
        prompt = f.read()

    pruned_metadata_str = prune_metadata_str(question, db_name)
    prompt = prompt.format(
        user_question=question, table_metadata_string=pruned_metadata_str
    )
    return prompt


def dynamic_num_beams(prompt: str, tokenizer, max_beams: int = 4) -> int:
    tokens = len(tokenizer.encode(prompt))
    print(tokens)
    if tokens <= 1024:
        return max_beams
    elif tokens <= 1536:
        return max_beams // 2
    else:
        return max_beams // 4


def get_tokenizer_model(model_name: Optional[str], adapter_path: Optional[str]):
    """
    Load a HuggingFace tokenizer and model.
    You may supply either a normal huggingface model name, or a peft adapter path.
    """
    if adapter_path is not None:
        print(f"Loading adapter model {adapter_path}")
        config = PeftConfig.from_pretrained(adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            use_cache=True,
            device_map="auto",
        )
        print(f"Loading adapter {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print(f"Merged adapter {adapter_path}")
    elif model_name is not None and "llama" in model_name:
        print(f"Loading Llama-based model {model_name}")
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
    else:
        print(f"Loading model {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )
    return tokenizer, model


def run_hf_eval(args):
    # get params from args
    questions_file = args.questions_file
    prompt_file_list = args.prompt_file
    num_questions = args.num_questions
    model_name = args.model
    adapter_path = args.adapter
    output_file_list = args.output_file

    if model_name is None and adapter_path is None:
        raise ValueError(
            "You must supply either a model name or an adapter path to run an evaluation."
        )

    print("questions prepared\nnow loading model...")
    # initialize tokenizer and model
    tokenizer, model = get_tokenizer_model(model_name, adapter_path)
    model.tie_weights()

    print("model loaded\nnow generating and evaluating predictions...")

    # from here, we generate and evaluate predictions
    # eos_token_id = tokenizer.convert_tokens_to_ids(["```"])[0]
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    for prompt_file, output_file in zip(prompt_file_list, output_file_list):
        print("preparing questions...")
        # get questions
        print(f"Using {num_questions} questions from {questions_file}")
        df = prepare_questions_df(questions_file, num_questions)
        # create a prompt for each question
        df["prompt"] = df[["question", "db_name"]].apply(
            lambda row: generate_prompt(
                prompt_file, row["question"], row["db_name"]
            ),
            axis=1,
        )

        total_tried = 0
        total_correct = 0
        output_rows = []

        with tqdm(total=len(df)) as pbar:
            for row in df.to_dict("records"):
                total_tried += 1
                start_time = time()
                num_beams = dynamic_num_beams(row["prompt"], tokenizer)
                # we set return_full_text to False so that we don't get the prompt text in the generated text
                # this simplifies our postprocessing to deal with just the truncation of the end of the query
                generated_query = (
                    pipe(
                        row["prompt"],
                        max_new_tokens=300,
                        do_sample=False,
                        num_beams=num_beams,
                        num_return_sequences=1,
                        return_full_text=False,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id,
                    )[0]["generated_text"]
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
        # get directory of output_file and create if not exist
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_df.to_csv(output_file, index=False, float_format="%.2f")
