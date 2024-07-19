#!/bin/zsh

model_names=("sqlcoder_8b_fullft_ds_012_llama3_old_join_hints_mgn1_b1_0900_b2_0990_steps_600")
db_type="postgres"
PORT=8083 # avoid 8081 as it's used by nginx
export CUDA_VISIBLE_DEVICES=1 # set gpu you want to use (just 1 will do)

# if db_type not postgres or sqlite, prompt_file should be prompts/prompt_cot.md else use prompts/prompt_cot_${dbtype}.md
if [ "$db_type" != "postgres" ] && [ "$db_type" != "sqlite" ]; then
  prompt_file="prompts/prompt_cot.md"
else
  prompt_file="prompts/prompt_cot_${db_type}.md"
fi

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
    # first, get the API up
    python3 utils/api_server.py --model "$model_path" --tensor-parallel-size 1 --dtype float16 --max-model-len 8192 --gpu-memory-utilization 0.90 --block-size 16 --disable-log-requests --port "${PORT}" &

    # run a loop to check if the http://localhost:8080/health endpoint returns a valid 200 result
    while true; do
      http_status=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}/health")
      if [ "$http_status" -eq 200 ]; then
        echo "API server is up and running"
        break
      else
        echo "Waiting for API server to be up..."
        sleep 1
      fi
    done

    # then run sql-eval
    python3 main.py -db "${db_type}" \
      -f "${prompt_file}" \
      -q "data/questions_gen_${db_type}.csv" "data/instruct_basic_${db_type}.csv" "data/instruct_advanced_${db_type}.csv" "data/idk.csv" \
      -o "results/${model_name}/c${checkpoint_num}_api_v1_cot.csv" "results/${model_name}/c${checkpoint_num}_api_basic_cot.csv" "results/${model_name}/c${checkpoint_num}_api_advanced_cot.csv" "results/${model_name}/c${checkpoint_num}_api_idk_cot.csv" \
      -g api \
      -b 1 \
      -c 0 \
      --api_url "http://localhost:${PORT}/generate" \
      --api_type "vllm" \
      -p 10 \
      --cot_table_alias "prealias" \
      --logprobs
    # finally, kill the api server
    pkill -9 -f "python3 utils/api_server.py.*--port ${PORT}"
  done
done
