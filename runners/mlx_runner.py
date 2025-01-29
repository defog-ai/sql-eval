import os
import pandas as pd
from tqdm import tqdm
from time import time

from runners.base_runner import BaseRunner
from utils.gen_prompt import generate_prompt
from utils.questions import prepare_questions_df
from utils.creds import db_creds_all
from eval.eval import compare_query_results
from utils.reporting import upload_results
from mlx_lm import load, generate


class MLXRunner(BaseRunner):
    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None

    def _initialize_model(self, model_path: str):
        """Initialize the MLX model."""
        self.model, self.tokenizer = load(model_path)

    def _call_llm(self, prompt, model_name, temperature=0.0):
        """Call MLX model."""
        # model_name is unused, as it's already loaded in _initialize_model
        text = generate(
            self.model, 
            self.tokenizer, 
            prompt=prompt, 
            max_tokens=512, 
            temp=0, 
            verbose=True
        )
        # Create a response object similar to other runners
        class MLXResponse:
            def __init__(self, content):
                self.content = content
                self.input_tokens = 0  # MLX doesn't provide token counts
                self.output_tokens = 0

        return MLXResponse(text)

    def _extract_query(self, response_content):
        """Extract SQL query from response."""
        return response_content.split(";")[0].split("```")[0].strip() + ";"

    def run_eval(self, args):
        """MLX-specific evaluation logic."""
        if not args.model:
            raise ValueError("Model path must be provided for MLX runner")

        self._initialize_model(args.model)
        print(f"Initialized MLX model from {args.model}")

        for questions_file, prompt_file, output_file in zip(
            args.questions_file, args.prompt_file, args.output_file
        ):
            print(f"Using prompt file {prompt_file}")
            print("Preparing questions...")
            print(
                f"Using {'all' if args.num_questions is None else args.num_questions} question(s) from {questions_file}"
            )
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
                    row.get("cot_pregen", False),
                    not args.use_private_data,
                    args.num_columns,
                    args.shuffle_metadata,
                ),
                axis=1,
            )

            # Process rows sequentially due to MLX's nature
            output_rows = []
            total_tried = total_correct = 0

            with tqdm(total=len(df)) as pbar:
                for row in df.to_dict("records"):
                    try:
                        result = self.process_row(row, None, args)  # None as model_name is unused
                        if result.get("correct", 0):
                            total_correct += 1
                        total_tried += 1
                        output_rows.append(result)
                    except Exception as e:
                        row["error_msg"] = f"PROCESSING ERROR: {str(e)}"
                        row["error_db_exec"] = 1
                        output_rows.append(row)
                    finally:
                        pbar.update(1)
                        pbar.set_description(
                            f"Correct so far: {total_correct}/{total_tried} ({100*total_correct/total_tried:.2f}%)"
                        )

            # Save results
            output_df = pd.DataFrame(output_rows)
            if "prompt" in output_df.columns:
                del output_df["prompt"]
                
            print(output_df.groupby("query_category")[["correct", "error_db_exec"]].mean())
            output_df = output_df.sort_values(by=["db_name", "query_category", "question"])
            
            # Save to file
            output_dir = os.path.dirname(output_file)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            try:
                output_df.to_csv(output_file, index=False, float_format="%.2f")
            except:
                output_df.to_pickle(output_file)
            print(f"Saved results to {output_file}")

            # Upload results if needed
            if args.upload_url is not None:
                with open(prompt_file, "r") as f:
                    prompt = f.read()
                upload_results(
                    results=output_df.to_dict("records"),
                    url=args.upload_url,
                    runner_type="mlx_runner",
                    prompt=prompt,
                    args=args,
                )


def run_mlx_eval(args):
    runner = MLXRunner()
    runner.run_eval(args)