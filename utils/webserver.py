# this is a Google cloud function for receiving the data from the web app and storing it in the database

import functions_framework
import psycopg2
import os

@functions_framework.http
def hello_http(request):
    request_json = request.get_json(force=True)
    results = request_json['results']
    run_id = request_json['run_id']
    timestamp = request_json['timestamp']
    runner_type = request_json['runner_type']
    prompt = request_json['prompt']
    prompt_id = request_json['prompt_id']
    model = request_json['model']
    num_beams = request_json['num_beams']
    db_type = request_json['db_type']
    gpu_name = request_json['gpu_name']
    gpu_memory = request_json['gpu_memory']
    gpu_driver_version = request_json['gpu_driver_version']
    gpu_cuda_version = request_json['gpu_cuda_version']
    num_gpus = request_json['num_gpus']
    conn = psycopg2.connect(
        dbname="sql_eval",
        user=os.environ.get("DB_USER"),
        password=os.environ.get("DB_PASSWORD"),
        host=os.environ.get("DB_HOST"),
        port=os.environ.get("DB_PORT"),
    )
    cur = conn.cursor()

    # add prompt to the prompts table if it doesn't exist
    cur.execute("SELECT * FROM prompts WHERE prompt_id = %s", (prompt_id,))
    if cur.fetchone() is None:
        cur.execute("INSERT INTO prompts (prompt_id, prompt) VALUES (%s, %s)", (prompt_id, prompt))

    for result in results:
        question = result['question']
        golden_query = result['query']
        db_name = result['db_name']
        query_category = result['query_category']
        generated_query = result['generated_query']
        error_msg = result['error_msg']
        exact_match = result['exact_match']
        correct = result['correct']
        error_db_exec = result['error_db_exec']
        latency_seconds = result['latency_seconds']
        tokens_used = result['tokens_used']

        cur.execute("INSERT INTO eval (run_id, question, golden_query, db_name, query_category, generated_query, error_msg, exact_match, correct, error_db_exec, latency_seconds, tokens_used, timestamp, runner_type, prompt_id, model, num_beams, db_type, gpu_name, gpu_memory, gpu_driver_version, gpu_cuda_version, num_gpus) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", (run_id, question, golden_query, db_name, query_category, generated_query, error_msg, exact_match, correct, error_db_exec, latency_seconds, tokens_used, timestamp, runner_type, prompt_id, model, num_beams, db_type, gpu_name, gpu_memory, gpu_driver_version, gpu_cuda_version, num_gpus))
    conn.commit()
    cur.close()
    conn.close()
    return "success"