# this is a Google cloud function for receiving the data from the web app and storing it in Postgres

import functions_framework
import psycopg2
import os

POSTGRES_DB = os.environ.get("POSTGRES_DB")
POSTGRES_HOST = os.environ.get("POSTGRES_HOST")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT")
POSTGRES_USER = os.environ.get("POSTGRES_USER")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD")


@functions_framework.http
def postgres(request):
    request_json = request.get_json(force=True)
    results = request_json["results"]
    run_id = request_json["run_id"]
    timestamp = request_json["timestamp"]
    runner_type = request_json["runner_type"]
    prompt = request_json["prompt"]
    prompt_id = request_json["prompt_id"]
    model = request_json["model"]
    num_beams = request_json["num_beams"]
    db_type = request_json["db_type"]
    gpu_name = request_json["gpu_name"]
    gpu_memory = request_json["gpu_memory"]
    gpu_driver_version = request_json["gpu_driver_version"]
    gpu_cuda_version = request_json["gpu_cuda_version"]
    num_gpus = request_json["num_gpus"]
    db_type = request_json.get("db_type", "bigquery")
    print(
        f"Received {len(results)} rows for run {run_id} at {timestamp} from {runner_type}"
    )
    conn = psycopg2.connect(
        dbname=POSTGRES_DB,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )
    print(f"Connected to the postgres db {POSTGRES_DB}")
    cur = conn.cursor()

    # add prompt to the prompts table if it doesn't exist
    cur.execute("SELECT * FROM prompt WHERE prompt_id = %s", (prompt_id,))
    if cur.fetchone() is None:
        cur.execute(
            "INSERT INTO prompt (prompt_id, prompt) VALUES (%s, %s)",
            (prompt_id, prompt),
        )
        print(f"Inserted prompt {prompt_id} into the prompts table")

    for result in results:
        question = result["question"]
        golden_query = result["query"]
        db_name = result["db_name"]
        query_category = result["query_category"]
        generated_query = result["generated_query"]
        error_msg = result["error_msg"]
        exact_match = bool(result["exact_match"])
        correct = bool(result["correct"])
        error_db_exec = bool(result["error_db_exec"])
        latency_seconds = result["latency_seconds"]
        tokens_used = result["tokens_used"]

        cur.execute(
            "INSERT INTO eval (run_id, question, golden_query, db_name, query_category, generated_query, error_msg, exact_match, correct, error_db_exec, latency_seconds, tokens_used, created_at, runner_type, prompt_id, model, num_beams, db_type, gpu_name, gpu_memory, gpu_driver_version, gpu_cuda_version, num_gpus) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                run_id,
                question,
                golden_query,
                db_name,
                query_category,
                generated_query,
                error_msg,
                exact_match,
                correct,
                error_db_exec,
                latency_seconds,
                tokens_used,
                timestamp,
                runner_type,
                prompt_id,
                model,
                num_beams,
                db_type,
                gpu_name,
                gpu_memory,
                gpu_driver_version,
                gpu_cuda_version,
                num_gpus,
            ),
        )
    print(f"Inserted {len(results)} rows into the postgres db {POSTGRES_DB}")
    conn.commit()
    cur.close()
    conn.close()
    return "success"
