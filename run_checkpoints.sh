#!/bin/zsh

model_names=("sqlcoder_8b_fullft_ds_003_llama3_mgn1_b1_0900_b2_0990")
PORT=8082 # avoid 8081 as it's used by nginx
export CUDA_VISIBLE_DEVICES=0 # set gpu you want to use (just 1 will do)

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
    python3 main.py -db postgres \
      -f prompts/prompt.md \
      -q "data/questions_gen_postgres.csv" "data/instruct_basic_postgres.csv" "data/instruct_advanced_postgres.csv" "data/idk.csv" \
      -o "results/${model_name}/c${checkpoint_num}_api_v1.csv" "results/${model_name}/c${checkpoint_num}_api_basic.csv" "results/${model_name}/c${checkpoint_num}_api_advanced.csv" "results/${model_name}/c${checkpoint_num}_api_idk.csv" \
      -g api \
      -b 1 \
      -c 0 \
      --api_url "http://localhost:${PORT}/generate" \
      --api_type "vllm" \
      -p 10
    # finally, kill the api server
    pkill -9 -f "python3 utils/api_server.py.*--port ${PORT}"
  done
done
