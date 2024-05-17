import os
import subprocess
import time

from transformers import AutoTokenizer

# Alternate version of gcs_eval if you're working with nested checkpoint folders
# with model weights instead of model weight folders directly

# edit these 4 paths as per your setup
# GCS_MODEL_DIR: gcs path where the models are stored.
# this should be the same as GCS_MODEL_DIR in model_uploader.py
# GCS_MODEL_EVAL_DIR: gcs path where the evaluated models will be shifted to
# LOCAL_MODEL_DIR: local path where the models will be downloaded
# SQL_EVAL_DIR: local path where the sql-eval repo is cloned
GCS_MODEL_DIR = "gs://defog-finetuning/fft"
GCS_MODEL_EVAL_DIR = "gs://defog-finetuning/fft_evaluated"
LOCAL_MODEL_DIR = os.path.expanduser("/models/fft")
SQL_EVAL_DIR = os.path.expanduser("~/sql-eval")
# edit the question files, prompt files and output files as you desire.
# they should have the same length, as they will be zipped and iterated through
# in the vllm runner.
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
os.chdir(SQL_EVAL_DIR)  # for executing sql-eval commands
# edit the run configs as per your requirements
NUM_BEAMS = 1
TOKENIZER_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


def check_and_save_tokenizer(dir: str):
    if not os.path.exists(os.path.join(dir, "tokenizer_config.json")):
        print(f"Saving tokenizer in {dir}")
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
        tokenizer.save_pretrained(dir)


def download_evaluate():
    while True:
        existing_models = (
            subprocess.run(["gsutil", "ls", GCS_MODEL_DIR], capture_output=True)
            .stdout.decode("utf-8")
            .split("\n")
        )
        existing_checkpoints = []
        for existing_model_folder in existing_models:
            results = (
                subprocess.run(
                    ["gsutil", "ls", existing_model_folder], capture_output=True
                )
                .stdout.decode("utf-8")
                .split("\n")
            )
            for path in results:
                if path.startswith(GCS_MODEL_DIR) and "checkpoint" in path:
                    existing_checkpoints.append(path)
        print("Existing checkpoints:")
        for ec in existing_checkpoints:
            print(ec)
        # sort existing checkpoints lexically
        existing_checkpoints.sort()
        for gcs_model_checkpoint_path in existing_checkpoints:
            run_name_checkpoint = gcs_model_checkpoint_path.replace(
                GCS_MODEL_DIR, ""
            ).strip(" /")
            if not run_name_checkpoint:
                print("No model found, skipping.")
                continue
            local_model_path = os.path.join(LOCAL_MODEL_DIR, run_name_checkpoint)
            run_name = run_name_checkpoint.split("/checkpoint-", 1)[0]
            print(f"Model name: {run_name_checkpoint}")
            if not os.path.exists(local_model_path):
                local_run_name_folder = os.path.join(LOCAL_MODEL_DIR, run_name)
                os.makedirs(local_run_name_folder, exist_ok=True)
                # download from gcs's checkpoint folder into a run name folder
                print(
                    f"Downloading from {gcs_model_checkpoint_path} to {local_run_name_folder}"
                )
                subprocess.run(
                    [
                        "gsutil",
                        "-m",
                        "cp",
                        "-r",
                        gcs_model_checkpoint_path,
                        local_run_name_folder,
                    ]
                )
            else:
                print(f"Model folder exists: {run_name_checkpoint}")
            check_and_save_tokenizer(local_model_path)
            try:
                # run evaluation
                # python3 main.py \
                #   -db postgres \
                #   -q data/instruct_basic_postgres.csv data/instruct_advanced_postgres.csv data/questions_gen_postgres.csv \
                #   -o "results/${run_name_checkpoint}_beam4_basic.csv" "results/${run_name_checkpoint}_beam4_advanced.csv" "results/${run_name_checkpoint}_beam4_v1.csv" \
                #   -g vllm \
                #   -b 4 \
                #   -c 0 \
                #   -f "prompts/prompt.md" \
                #   -m "/models/fsdp/${run_name_checkpoint}"
                question_files = [
                    "data/instruct_basic_postgres.csv",
                    "data/instruct_advanced_postgres.csv",
                    "data/questions_gen_postgres.csv",
                ]
                prompt_file = "prompts/prompt.md"
                output_files = [
                    f"results/{run_name_checkpoint}_beam{NUM_BEAMS}_basic.csv",
                    f"results/{run_name_checkpoint}_beam{NUM_BEAMS}_advanced.csv",
                    f"results/{run_name_checkpoint}_beam{NUM_BEAMS}_v1.csv",
                ]
                os.makedirs(os.path.join("results", run_name), exist_ok=True)
                subprocess.run(
                    [
                        "python3",
                        "main.py",
                        "-db",
                        "postgres",
                        "-q",
                        *question_files,
                        "-o",
                        *output_files,
                        "-g",
                        "vllm",
                        "-b",
                        str(NUM_BEAMS),
                        "-c",
                        "0",
                        "-f",
                        prompt_file,
                        "-m",
                        local_model_path,
                        "-bs",
                        "200",
                    ],
                    check=True,
                )
                # make model directory in gcs
                subprocess.run(
                    [
                        "gsutil",
                        "mkdir",
                        f"{GCS_MODEL_EVAL_DIR}/{run_name}",
                    ]
                )
                # move the model to the evaluated directory once evaluated successfully
                subprocess.run(
                    [
                        "gsutil",
                        "-m",
                        "mv",
                        gcs_model_checkpoint_path,
                        f"{GCS_MODEL_EVAL_DIR}/{run_name}",
                    ],
                    check=True,
                )
                subprocess.run(["rm", "-rf", local_model_path], check=True)
            except Exception as e:
                print(f"Error in evaluation: {e}")
                exit(1)
        time.sleep(10)


if __name__ == "__main__":
    download_evaluate()
