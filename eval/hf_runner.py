import json
import os
from typing import Optional

from eval.eval import compare_query_results
import pandas as pd
import torch
import traceback
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)
from utils.pruning import prune_metadata_str
from utils.questions import prepare_questions_df
from utils.creds import db_creds_all
from tqdm import tqdm
from psycopg2.extensions import QueryCanceledError
from time import time
import gc
from peft import PeftModel, PeftConfig
from utils.reporting import upload_results

device_map = "mps" if torch.backends.mps.is_available() else "auto"


def generate_prompt(
    prompt_file, question, db_name, instructions="", k_shot_prompt="", glossary="", table_metadata_string="", public_data=True
):
    with open(prompt_file, "r") as f:
        prompt = f.read()
    question_instructions = question + " " + instructions

    if table_metadata_string == "":
        pruned_metadata_str = prune_metadata_str(
            question_instructions, db_name, public_data
        )
    else:
        pruned_metadata_str = table_metadata_string

    prompt = prompt.format(
        user_question=question,
        instructions=instructions,
        table_metadata_string=pruned_metadata_str,
        k_shot_prompt=k_shot_prompt,
        glossary=glossary,
    )
    return prompt


def dynamic_num_beams(prompt: str, tokenizer, max_beams: int = 4) -> int:
    tokens = len(tokenizer.encode(prompt))
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
            device_map=device_map,
        )
        print(f"Loading adapter {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print(f"Merged adapter {adapter_path}")
    else:
        print(f"Loading model {model_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except:
            tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-34b-hf")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map=device_map,
        )
    return tokenizer, model


def run_hf_eval(args):
    # get params from args
    questions_file = args.questions_file
    prompt_file_list = args.prompt_file
    num_questions = args.num_questions
    public_data = not args.use_private_data
    model_name = args.model
    adapter_path = args.adapter
    output_file_list = args.output_file
    k_shot = args.k_shot
    db_type = args.db_type

    if model_name is None and adapter_path is None:
        raise ValueError(
            "You must supply either a model name or an adapter path to run an evaluation."
        )

    print(f"Questions prepared\nNow loading model...")
    # initialize tokenizer and model
    tokenizer, model = get_tokenizer_model(model_name, adapter_path)
    model.tie_weights()

    print("model loaded\nnow generating and evaluating predictions...")

    # from here, we generate and evaluate predictions
    # eos_token_id = tokenizer.convert_tokens_to_ids(["```"])[0]
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    support_beam_search = True

    # check that the model supports beam search with a try except statement
    try:
        pipe("Hi", num_beams=2, do_sample=False, max_new_tokens=5)
    except AttributeError as e:
        error_trace = traceback.format_exception(type(e), e, e.__traceback__)
        support_beam_search = (
            len([line for line in error_trace if "self.beam_search" in line]) == 0
        )
        if not support_beam_search:
            print(
                "WARNING: This model does not support beam search. will use num_beams=1"
            )
        else:
            raise e

    # get questions
    print("Preparing questions...")
    print(
        f"Using {'all' if num_questions is None else num_questions} question(s) from {questions_file}"
    )
    df = prepare_questions_df(questions_file, db_type, num_questions, k_shot)

    for prompt_file, output_file in zip(prompt_file_list, output_file_list):
        # create a prompt for each question
        df["prompt"] = df[
            ["question", "db_name", "instructions", "k_shot_prompt", "glossary", "table_metadata_string"]
        ].apply(
            lambda row: generate_prompt(
                prompt_file,
                row["question"],
                row["db_name"],
                row["instructions"],
                row["k_shot_prompt"],
                row["glossary"],
                row["table_metadata_string"],
                public_data,
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

                num_beams = 1
                if support_beam_search:
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
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                end_time = time()

                row["generated_query"] = generated_query
                row["latency_seconds"] = end_time - start_time
                golden_query = row["query"]
                db_name = row["db_name"]
                db_type = row["db_type"]
                question = row["question"]
                query_category = row["query_category"]
                exact_match = correct = 0
                db_creds = db_creds_all[db_type]

                try:
                    exact_match, correct = compare_query_results(
                        query_gold=golden_query,
                        query_gen=generated_query,
                        db_name=db_name,
                        db_type=db_type,
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

        results = output_df.to_dict("records")
        # upload results
        with open(prompt_file, "r") as f:
            prompt = f.read()
        if args.upload_url is not None:
            upload_results(
                results=results,
                url=args.upload_url,
                runner_type="hf_runner",
                prompt=prompt,
                args=args,
            )
