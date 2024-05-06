import os
import subprocess
import time
from typing import List

from eval.vllm_runner import run_vllm_eval

# edit these 4 paths as per your setup
# GCS_MODEL_DIR: gcs path where the models are stored.
# this should be the same as GCS_MODEL_DIR in model_uploader.py
# GCS_MODEL_EVAL_DIR: gcs path where the evaluated models will be shifted to
# LOCAL_MODEL_DIR: local path where the models will be downloaded
# SQL_EVAL_DIR: local path where the sql-eval repo is cloned
GCS_MODEL_DIR = "gs://defog-finetuning/fsdp_wrong_sql_eval"
GCS_MODEL_EVAL_DIR = "gs://defog-finetuning/fsdp_evaluated"
LOCAL_MODEL_DIR = os.path.expanduser("/models/fsdp")
SQL_EVAL_DIR = os.path.expanduser("~/sql-eval")
# edit the question files, prompt files and output files as you desire.
# they should have the same length, as they will be zipped and iterated through
# in the vllm runner.
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
os.chdir(SQL_EVAL_DIR)  # for executing sql-eval commands
# edit the run configs as per your requirements
NUM_BEAMS = 1


def download_evaluate():
    while True:
        existing_models = (
            subprocess.run(["gsutil", "ls", GCS_MODEL_DIR], capture_output=True)
            .stdout.decode("utf-8")
            .split("\n")
        )
        for gcs_model_path in existing_models:
            model_name = (
                gcs_model_path.replace(GCS_MODEL_DIR, "").replace("/", "").strip()
            )
            if not model_name:
                continue
            local_model_path = os.path.join(LOCAL_MODEL_DIR, model_name)
            if not os.path.exists(local_model_path):
                print(f"Downloading model: {model_name}")
                # download from gcs
                subprocess.run(
                    ["gsutil", "-m", "cp", "-r", gcs_model_path, LOCAL_MODEL_DIR]
                )
            else:
                print(f"Model folder exists: {model_name}")
            try:
                # run evaluation
                # python3 main.py \
                #   -db postgres \
                #   -q data/instruct_basic_postgres.csv data/instruct_advanced_postgres.csv data/questions_gen_postgres.csv \
                #   -o "results/${model_name}_beam4_basic.csv" "results/${model_name}_beam4_advanced.csv" "results/${model_name}_beam4_v1.csv" \
                #   -g vllm \
                #   -b 4 \
                #   -c 0 \
                #   -f "prompts/prompt.md" \
                #   -m "/models/fsdp/${model_name}"
                question_files = [
                    "data/instruct_basic_postgres.csv",
                    "data/instruct_advanced_postgres.csv",
                    "data/questions_gen_postgres.csv",
                ]
                prompt_file = "prompts/prompt.md"
                output_files = [
                    f"results/{model_name}_beam{NUM_BEAMS}_basic.csv",
                    f"results/{model_name}_beam{NUM_BEAMS}_advanced.csv",
                    f"results/{model_name}_beam{NUM_BEAMS}_v1.csv",
                ]
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
                        "16",
                    ],
                    check=True,
                )
                # move the model to the evaluated directory once evaluated successfully
                subprocess.run(
                    ["gsutil", "-m", "mv", gcs_model_path, GCS_MODEL_EVAL_DIR],
                    check=True,
                )
                subprocess.run(["rm", "-rf", local_model_path], check=True)
            except Exception as e:
                print(f"Error in evaluation: {e}")
                exit(1)
        time.sleep(10)


if __name__ == "__main__":
    download_evaluate()
