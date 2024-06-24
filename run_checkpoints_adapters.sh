#!/bin/zsh

model_names=("sqlcoder_8b_bf16_r64_ds_013_sqlite_600_b24_lr8e-5")
base_model_path="defog/sqlcoder-8b-padded-sorry"
db_type="sqlite"
PORT=8084 # avoid 8081 as it's used by nginx
export CUDA_VISIBLE_DEVICES=1 # set gpu you want to use (just 1 will do)
preprocess_adapters=true # set to false if you have already preprocessed the adapters
cot_table_alias=true # set to true if you want to use the cot_table_alias prompt in evals

# check that the base model was trained on cot data otherwise print a warning
if [[ ! $base_model_path == *"cot"* ]] && [[ $cot_table_alias == true ]]; then
  echo "WARNING: Base model was not trained on 'cot' data. This may lead to less than optimal results"
fi
for model_name in "${model_names[@]}"; do
  # list the folder names in /models/combined/${model_name}
  adapter_dir="${HOME}/finetuning/models/${model_name}"
  echo "Adapter directory: ${adapter_dir}"
  checkpoints=($(ls $adapter_dir))
  echo "Checkpoints: ${checkpoints}"
  if [ "$preprocess_adapters" = true ]; then
    # Preprocess the adapters
    for checkpoint in "${checkpoints[@]}"; do
      # skip if does not start with "checkpoint-"
      if [[ ! $checkpoint == checkpoint-* ]]; then
        continue
      fi
      checkpoint_num=$(echo $checkpoint | cut -d'-' -f2)
      echo "Preprocessing adapter ${model_name}/checkpoint-${checkpoint_num}"
      python3 ${HOME}/finetuning/preprocess_adapters.py -f ${adapter_dir}/checkpoint-${checkpoint_num}/adapter_model.safetensors
    done
  else
    echo "Skipping preprocessing adapters..."
  fi
 
  # Loop over checkpoints to run sql-eval
  for checkpoint in "${checkpoints[@]}"; do
    # skip if does not start with "checkpoint-"
    if [[ ! $checkpoint == checkpoint-* ]]; then
      continue
    fi
    checkpoint_num=$(echo $checkpoint | cut -d'-' -f2)
    echo "Running adapter ${model_name} checkpoint ${checkpoint_num}"
    # first, get the API up
    python3 utils/api_server.py --model "$base_model_path" --tensor-parallel-size 1 --dtype float16 --max-model-len 4096 --gpu-memory-utilization 0.90 --block-size 16 --disable-log-requests --port "${PORT}" --enable-lora --max-lora-rank 64 &

    # run a loop to check if the http://localhost:8084/health endpoint returns a valid 200 result
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
    if [ "$cot_table_alias" = true ]; then
      python3 main.py -db ${db_type} \
        -f prompts/prompt_cot.md \
        -q "data/instruct_basic_${db_type}.csv" "data/instruct_advanced_${db_type}.csv" "data/questions_gen_${db_type}.csv" "data/idk.csv" \
        -o "results/${model_name}/${model_name}_c${checkpoint_num}_${db_type}_api_cot_basic.csv" "results/${model_name}/${model_name}_c${checkpoint_num}_${db_type}_api_cot_advanced.csv" "results/${model_name}/${model_name}_c${checkpoint_num}_${db_type}_api_cot_v1.csv" "results/${model_name}/${model_name}_c${checkpoint_num}_${db_type}_api_cot_idk.csv" \
        -g api \
        -b 1 \
        -c 0 \
        --api_url "http://localhost:${PORT}/generate" \
        --api_type "vllm" \
        -p 10 \
        -a ${adapter_dir}/checkpoint-${checkpoint_num}\
        --cot_table_alias "prealias" \
        --logprobs 
    else
      python3 main.py -db ${db_type} \
      -f prompts/prompt.md \
      -q "data/instruct_basic_${db_type}.csv" "data/instruct_advanced_${db_type}.csv" "data/questions_gen_${db_type}.csv" "data/idk.csv" \
      -o "results/${model_name}/${model_name}_c${checkpoint_num}_${db_type}_api_basic.csv" "results/${model_name}/${model_name}_c${checkpoint_num}_${db_type}_api_advanced.csv" "results/${model_name}/${model_name}_c${checkpoint_num}_${db_type}_api_v1.csv" "results/${model_name}/${model_name}_c${checkpoint_num}_${db_type}_api_idk.csv" \
      -g api \
      -b 1 \
      -c 0 \
      --api_url "http://localhost:${PORT}/generate" \
      --api_type "vllm" \
      -p 10 \
      -a ${adapter_dir}/checkpoint-${checkpoint_num}\
      --logprobs
    fi
    # finally, kill the api server
    pkill -9 -f "python3 utils/api_server.py.*--port ${PORT}"
  done
done

# pass all the model_names to the python script
python3 analyze_results_and_post_to_slack.py -m "${model_names[@]}"