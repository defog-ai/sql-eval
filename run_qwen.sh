export db_type="postgres"
export prompt_file="prompts/prompt_qwen.json"
export model_name="qwen"
export PORT=8000

# assume you already have the vllm server running
# vllm serve "$model_name" --port 8000

if [[ "$1" == "--thinking" ]]; then
    echo "Running sql-eval on $model_name with thinking tokens"
    python3 main.py -db "${db_type}" \
        -f "${prompt_file}" \
        -q "data/questions_gen_${db_type}.csv" "data/instruct_basic_${db_type}.csv" "data/instruct_advanced_${db_type}.csv" \
        -o "results/${model_name}/openai_api_v1.csv" "results/${model_name}/openai_api_basic.csv" "results/${model_name}/openai_api_advanced.csv" \
        -g api \
        -m "Qwen/Qwen3-4B" \
        -b 1 \
        -c 0 \
        --thinking \
        --api_url "http://localhost:${PORT}/v1/chat/completions" \
        --api_type "openai" \
        -p 10
else
    echo "Running sql-eval on $model_name without generating thinking tokens"
    python3 main.py -db "${db_type}" \
        -f "${prompt_file}" \
        -q "data/questions_gen_${db_type}.csv" "data/instruct_basic_${db_type}.csv" "data/instruct_advanced_${db_type}.csv" \
        -o "results/${model_name}/openai_api_v1.csv" "results/${model_name}/openai_api_basic.csv" "results/${model_name}/openai_api_advanced.csv" \
        -g api \
        -m "Qwen/Qwen3-4B" \
        -b 1 \
        -c 0 \
        --api_url "http://localhost:${PORT}/v1/chat/completions" \
        --api_type "openai" \
        -p 10
fi