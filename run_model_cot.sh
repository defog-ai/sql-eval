#!/bin/zsh

# model_dir="${HOME}/models"
# parse in model_names from args to script
model_names=("$@")
db_type="postgres"
# if model_names is empty, print and exit
if [ -z "$model_names" ]; then
  echo "No model names provided"
  exit 1
fi
PORT=8084 # avoid 8081 as it's used by nginx
export CUDA_VISIBLE_DEVICES=0 # set gpu you want to use (just 1 will do)

# if db_type not postgres or sqlite, prompt_file should be prompts/prompt_cot.md else use prompts/prompt_cot_${dbtype}.md
if [ "$db_type" != "postgres" ] && [ "$db_type" != "sqlite" ]; then
  prompt_file="prompts/prompt_cot.md"
else
  prompt_file="prompts/prompt_cot_${db_type}.md"
fi

# Loop over model names
for model_name in "${model_names[@]}"; do

  echo "Running model ${model_name}"
  # first, get the API up
  python3 utils/api_server.py --model "${model_name}" --tensor-parallel-size 1 --dtype float16 --max-model-len 16384 --gpu-memory-utilization 0.90 --block-size 16 --disable-log-requests --port "${PORT}" &

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
    -o "results/${model_name}/api_v1_cot.csv" "results/${model_name}/api_basic_cot.csv" "results/${model_name}/api_advanced_cot.csv" "results/${model_name}/api_idk_cot.csv" \
    -g api \
    -b 1 \
    -c 0 \
    --api_url "http://localhost:${PORT}/generate" \
    --api_type "vllm" \
    -p 10 \
    --logprobs
  # finally, kill the api server
  pkill -9 -f "python3 utils/api_server.py.*--port ${PORT}"

done
