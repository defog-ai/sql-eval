import os
from typing import List
import sqlparse
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import pandas as pd
import time
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from runners.base_runner import BaseRunner
from utils.gen_prompt import generate_prompt
from utils.questions import prepare_questions_df
from utils.creds import db_creds_all
from eval.eval import compare_query_results
from utils.reporting import upload_results


class VLLMRunner(BaseRunner):
    def __init__(self):
        super().__init__()
        self.llm = None
        self.tokenizer = None
        self.sampling_params = None

    def _initialize_model(self, args):
        """Initialize VLLM model with appropriate parameters."""
        model_name = args.model
        enable_lora = True if args.adapter else False
        lora_request = LoRARequest("sql_adapter", 1, args.adapter) if args.adapter else None
        self.lora_request = lora_request

        print(f"Preparing {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        if not args.quantized:
            self.llm = LLM(
                model=model_name,
                tensor_parallel_size=1,
                enable_lora=enable_lora,
                max_model_len=4096,
                max_lora_rank=64,
            )
        else:
            self.llm = LLM(
                model=model_name,
                tensor_parallel_size=1,
                quantization="AWQ",
                enable_lora=enable_lora,
                max_model_len=4096,
                max_lora_rank=64,
            )

        self.sampling_params = SamplingParams(
            n=1,
            best_of=args.num_beams,
            use_beam_search=args.num_beams != 1,
            stop_token_ids=[self.tokenizer.eos_token_id],
            max_tokens=1000,
            temperature=0,
        )

    def _extract_query(self, response_content):
        """Extract SQL query from response."""
        try:
            # Try to extract query from code blocks
            query = response_content.split("```sql", 1)[-1].split("```")[0].strip()
            return query.split(";")[0].strip() + ";"
        except:
            # Fallback to extract anything that looks like SQL
            return response_content.split(";")[0].strip() + ";"

    def _process_batch(self, batch, args):
        """Process a batch of questions using VLLM."""
        prompts = batch["prompt"].tolist()
        print(f"Generating completions for {len(prompts)} prompts")
        
        # Tokenize prompts
        prompt_tokens = []
        prompt_token_sizes = []
        for prompt in prompts:
            token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            if token_ids[0] != self.tokenizer.bos_token_id:
                token_ids = [self.tokenizer.bos_token_id] + token_ids
            prompt_tokens.append(token_ids)
            prompt_token_sizes.append(len(token_ids))
        
        print(f"Average prompt size: {sum(prompt_token_sizes)/len(prompt_token_sizes):.0f}")
        
        # Generate completions
        start_time = time.time()
        outputs = self.llm.generate(
            sampling_params=self.sampling_params,
            prompt_token_ids=prompt_tokens,
            use_tqdm=False,
            lora_request=self.lora_request,
        )
        time_taken = time.time() - start_time
        print(f"Generated {len(outputs)} completions in {time_taken:.2f} seconds")

        # Process results
        results = []
        for row, output in zip(batch.to_dict("records"), outputs):
            generated_query = self._extract_query(output.outputs[0].text)
            generated_query = sqlparse.format(
                generated_query, keyword_case="upper", strip_whitespace=True
            )
            
            row["generated_query"] = generated_query
            row["tokens_used"] = len(output.outputs[0].token_ids)
            row["latency_seconds"] = time_taken / len(batch)

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
            except Exception as e:
                row["error_db_exec"] = 1
                row["error_msg"] = f"QUERY EXECUTION ERROR: {e}"

            results.append(row)
        return results

    def run_eval(self, args):
        """VLLM-specific evaluation logic with batching."""
        self._initialize_model(args)

        for questions_file, prompt_file, output_file in zip(
            args.questions_file, args.prompt_file, args.output_file
        ):
            print(f"Using prompt file {prompt_file}")
            print("Preparing questions...")
            print(f"Using {'all' if args.num_questions is None else args.num_questions} question(s) from {questions_file}")
            df = prepare_questions_df(
                questions_file, args.db_type, args.num_questions, args.k_shot, args.cot_table_alias
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
                    row.get("cot_pregen", ""),
                    not args.use_private_data,
                    args.num_columns,
                    args.shuffle_metadata,
                ),
                axis=1,
            )
            print(f"Prepared {len(df)} question(s) from {questions_file}")

            # Process in batches
            def chunk_dataframe(df, chunk_size):
                df_chunks = []
                for i in range(0, len(df), chunk_size):
                    df_i = df.iloc[i : min(i + chunk_size, len(df))]
                    print(f"Chunk {i//chunk_size+1}/{len(df)//chunk_size+1} with {len(df_i)} questions")
                    df_chunks.append(df_i)
                return df_chunks

            df_chunks = chunk_dataframe(df, args.batch_size)
            all_results = []
            total_tried = total_correct = 0

            print("Generating completions")
            for batch in (pbar := tqdm(df_chunks, total=len(df_chunks))):
                batch_results = self._process_batch(batch, args)
                all_results.extend(batch_results)
                
                # Update progress stats
                batch_correct = sum(1 for r in batch_results if r.get("correct", 0))
                total_correct += batch_correct
                total_tried += len(batch)
                pbar.set_description(
                    f"Correct so far: {total_correct}/{total_tried} ({100*total_correct/total_tried:.2f}%)"
                )

            # Save results
            results_df = pd.DataFrame(all_results)
            if "prompt" in results_df.columns:
                del results_df["prompt"]
            
            print(results_df.groupby("query_category")[["exact_match", "correct"]].mean())
            results_df = results_df.sort_values(by=["db_name", "query_category", "question"])
            print(f"Average tokens generated: {results_df['tokens_used'].mean():.1f}")
            
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
                    runner_type="vllm_runner",
                    prompt=prompt,
                    args=args,
                )


def run_vllm_eval(args):
    runner = VLLMRunner()
    runner.run_eval(args)