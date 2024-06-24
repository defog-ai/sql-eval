# this is a Google cloud function for receiving the data from the web app and storing it in the database
# to launch the cloud function, run the following command in the terminal:
# gcloud functions deploy record-eval --runtime python10 --trigger-http --allow-unauthenticated

import functions_framework
from google.cloud import storage
import json

BUCKET_NAME = "YOUR-BUCKET-NAME"


@functions_framework.http
def hello_http(request):
    request_json = request.get_json(silent=True)
    results = request_json["results"]
    run_name = request_json["run_name"]
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(run_name + ".json")
    blob.upload_from_string(json.dumps(results))
    return "success"
