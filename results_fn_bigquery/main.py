# this is a Google cloud function for receiving the data from the web app and storing it in Bigquery

import functions_framework
from google.oauth2 import service_account
import pandas as pd
import pandas_gbq
import os

BQ_PROJECT = os.environ.get("BQ_PROJECT")
BQ_TABLE = os.environ.get("BQ_TABLE")

# authenticate using service account's json credentials
credentials_path = os.environ.get("CREDENTIALS_PATH")
print(f"CREDENTIALS_PATH: {credentials_path}")
credentials = service_account.Credentials.from_service_account_file(credentials_path)
print(f"Credentials: {credentials}")


@functions_framework.http
def bigquery(request):
    request_json = request.get_json(force=True)
    results = request_json["results"]
    run_id = request_json["run_id"]
    run_time = pd.to_datetime(request_json["timestamp"])
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
    run_args = request_json["run_args"]

    if len(results) == 0:
        return "no results to write"

    df = pd.DataFrame(results)
    df["run_time"] = run_time
    print(f"results:\n{results}")
    print(f"df:\n{df}")
    # add other metadata to the dataframe
    run_args["run_id"] = run_id
    run_args["runner_type"] = runner_type
    run_args["prompt"] = prompt
    run_args["prompt_id"] = prompt_id
    run_args["model"] = model
    run_args["num_beams"] = num_beams
    run_args["db_type"] = db_type
    run_args["gpu_name"] = gpu_name
    run_args["gpu_memory"] = gpu_memory
    run_args["gpu_driver_version"] = gpu_driver_version
    run_args["gpu_cuda_version"] = gpu_cuda_version
    run_args["num_gpus"] = num_gpus
    df["run_params"] = run_args
    print(f"df with run_params:\n{df}")
    # write to bigquery
    pandas_gbq.to_gbq(
        dataframe=df,
        destination_table=BQ_TABLE,
        project_id=BQ_PROJECT,
        if_exists="append",
        progress_bar=False,
        credentials=credentials,
    )
    return "success"
