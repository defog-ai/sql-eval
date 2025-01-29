import os
from typing import Optional
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)
from tqdm import tqdm
from psycopg2.extensions import QueryCanceledError
import gc

from runners.base_runner import BaseRunner
from utils.gen_prompt import generate_prompt
from utils.questions import prepare_questions_df
from utils.creds import db_creds_all
from eval.eval import compare_query_results
from utils.reporting import upload_results

device_map = "mps" if torch.backends.mps.is_available() else "auto"


class HFRunner(BaseRunner):
    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.model = None
        self.pipe = None

    def _initialize_model(
        self, model_name: Optional[str], adapter_path: Optional[str], batch_size: int
    ):
        """Load a HuggingFace tokenizer and model."""
        if adapter_path is not None:
            from peft import PeftModel, PeftConfig

            print(f"Loading adapter model {adapter_path}")
            config = PeftConfig.from_pretrained(adapter_path)
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.base_model_name_or_path
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                use_cache=True,
                device_map=device_map,
            )
            print(f"Loading adapter {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model = self.model.merge_and_unload()
            print(f"Merged adapter {adapter_path}")
        else:
            print(f"Loading model {model_name}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            except:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "meta-llama/Meta-Llama-3-8B-Instruct"
                )

        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if model_name and "8b" in model_name.lower():
            # do this since it doesn't seem to have been done by default
            self.tokenizer.padding_side = "left"

        if not self.model:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map=device_map,
            )

        self.model.tie_weights()
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            batch_size=batch_size,
        )

    def _extract_query(self, generated_text, prompt):
        """Extract SQL query from response based on prompt format."""
        if "[SQL]" not in prompt:
            return generated_text.split("```")[0].split(";")[0].strip() + ";"
        else:
            return generated_text.split("[/SQL]")[0].split(";")[0].strip() + ";"

    def _process_batch(self, batch, args):
        """Process a batch of questions using HF pipeline."""
        prompts = batch["prompt"].tolist()
        generated_queries = self.pipe(
            prompts,
            max_new_tokens=600,
            do_sample=False,
            num_beams=args.num_beams,
            num_return_sequences=1,
            return_full_text=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            temperature=None,
            top_p=None,
        )

        # Clean up GPU memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        results = []
        for row, result in zip(batch.to_dict("records"), generated_queries):
            generated_query = self._extract_query(
                result[0]["generated_text"], row["prompt"]
            )

            # More GPU cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            row["generated_query"] = generated_query
            row["latency_seconds"] = (
                None  # HF pipeline doesn't provide per-item latency
            )

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
                row["exact_match"] = int(exact_match)
                row["correct"] = int(correct)
                row["error_msg"] = ""
            except QueryCanceledError as e:
                row["timeout"] = 1
                row["error_msg"] = f"QUERY EXECUTION TIMEOUT: {e}"
            except Exception as e:
                row["error_db_exec"] = 1
                row["error_msg"] = f"QUERY EXECUTION ERROR: {e}"

            results.append(row)
        return results

    def run_eval(self, args):
        """HF-specific evaluation logic with batching."""
        if args.model is None and args.adapter is None:
            raise ValueError(
                "You must supply either a model name or an adapter path to run an evaluation."
            )

        self._initialize_model(args.model, args.adapter, args.batch_size)

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
            # Create prompts for all questions
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
                    not args.use_private_data,
                    args.num_columns,
                    args.shuffle_metadata,
                ),
                axis=1,
            )

            # Process in batches
            def chunk_dataframe(df, chunk_size):
                for i in range(0, len(df), chunk_size):
                    yield df[i : min(i + chunk_size, len(df))]

            df_chunks = list(chunk_dataframe(df, args.batch_size))
            all_results = []
            total_tried = total_correct = 0

            with tqdm(total=len(df)) as pbar:
                for batch in df_chunks:
                    batch_results = self._process_batch(batch, args)
                    all_results.extend(batch_results)

                    # Update progress stats
                    batch_correct = sum(1 for r in batch_results if r.get("correct", 0))
                    total_correct += batch_correct
                    total_tried += len(batch)
                    pbar.update(len(batch))
                    pbar.set_description(
                        f"Correct so far: {total_correct}/{total_tried} ({100*total_correct/total_tried:.2f}%)"
                    )

            # Save results
            results_df = pd.DataFrame(all_results)
            if "prompt" in results_df.columns:
                del results_df["prompt"]

            print(
                results_df.groupby("query_category")[
                    ["correct", "error_db_exec"]
                ].mean()
            )
            results_df = results_df.sort_values(
                by=["db_name", "query_category", "question"]
            )

            # Save to file
            output_dir = os.path.dirname(output_file)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            results_df.to_csv(output_file, index=False, float_format="%.2f")
            print(f"Saved results to {output_file}")

            # Upload results if needed
            if args.upload_url is not None:
                with open(prompt_file, "r") as f:
                    prompt = f.read()
                upload_results(
                    results=results_df.to_dict("records"),
                    url=args.upload_url,
                    runner_type="hf_runner",
                    prompt=prompt,
                    args=args,
                )


def run_hf_eval(args):
    runner = HFRunner()
    runner.run_eval(args)
