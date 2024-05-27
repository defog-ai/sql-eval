#!/bin/zsh

model_names=("sqlcoder_8b_fullft_ds_009_llama3_mgn1_b1_0900_b2_0990")

# Loop over model names
for model_name in "${model_names[@]}"; do
  # list the folder names in /models/combined/${model_name}
  model_dir="/workspace/finetuning/models/${model_name}"
  echo "Model directory: ${model_dir}"
  checkpoints=($(ls $model_dir))
  echo "Checkpoints: ${checkpoints}"
  # Loop over checkpoints
  for checkpoint in "${checkpoints[@]}"; do
    # skip if does not start with "checkpoint-"
    if [[ ! $checkpoint == checkpoint-* ]]; then
      continue
    fi
    model_path="${model_dir}/${checkpoint}"
    checkpoint_num=$(echo $checkpoint | cut -d'-' -f2)
    echo "Running model ${model_name} checkpoint ${checkpoint_num}"
    python -W ignore main.py \
      -db postgres \
      -q data/instruct_basic_postgres.csv data/instruct_advanced_postgres.csv data/questions_gen_postgres.csv \
      -o "results/${model_name}/c${checkpoint_num}_basic_api_a100.csv" "results/${model_name}/c${checkpoint_num}_advanced_api_a100.csv" "results/${model_name}/c${checkpoint_num}_v1_api_a100.csv" \
      -g vllm \
      -m "$model_path" \
      -c 0 \
      -b 1 \
      -bs 10 \
      -f prompts/prompt.md
  done
done
