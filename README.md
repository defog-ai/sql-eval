# SQL Generation Evaluation

[![tests](https://github.com/defog-ai/sql-generation-evaluation/actions/workflows/main.yml/badge.svg)](https://github.com/defog-ai/sql-generation-evaluation/actions/workflows/main.yml)

This repository contains the code that Defog uses for the evaluation of generated SQL. It's based off the schema from the [Spider](https://github.com/taoyds/spider), but with a new set of hand-selected questions and queries grouped by query category. For an in-depth look into our process of creating this evaluation approach, see [this](https://defog.ai/blog/open-sourcing-sqleval/).

## Introduction

Our testing procedure comprises the following steps. For each question/query pair:

1. We generate a SQL query (possibly from an LLM).
2. We run both the "gold" query and the generated query on their respective database to obtain 2 dataframes with the results.
3. We compare the 2 dataframes using an "exact" and a "subset" match. TODO add link to blogpost.
4. We log these alongside other metrics of interest (e.g. tokens used, latency) and aggregate the results for reporting.

## Getting Started

This is a comprehensive set of instructions that assumes basic familiarity with the command line, Docker, running SQL queries on a database, and common Python data manipulation libraries (e.g. pandas).

### Install Dependencies

Firstly, clone the repository where we store our database data and schema. Install all Python libraries listed in the `requirements.txt` file. You would also need to download a spacy model if you're using the NER heuristic for our [metadata-pruning method](https://github.com/defog-ai/sql-eval/blob/main/utils/pruning.py) (set by any values of the `c` parameter that is  more than 0, more below). Finally, install the library.

```bash
git clone https://github.com/defog-ai/defog-data.git
cd defog-data
pip install -r requirements.txt
pip install -e .
```

### Start Postgres Instance

Next, you would need to set up the databases that the queries are executed on. We use Postgres here, since it is the most common OSS database with the widest distribution and usage in production. In addition, we would recommend using Docker to do this, as it is the easiest way to get started. You can install Docker [here](https://docs.docker.com/get-docker/).

Once you have Docker installed, you can create the Docker container and start the Postgres database using the following commands. We recommend mounting a volume on `data/postgres` to persist the data, as well as `data/export` to make it easier to import the data. To create the container, run:

```bash
mkdir data/postgres data/export
docker create --name postgres-sql-eval -e POSTGRES_PASSWORD=postgres -p 5432:5432 -v $(pwd)/data/postgres:/var/lib/postgresql/data -v $(pwd)/data/export:/export postgres:16-alpine
```

To start the container, run:

```bash
docker start postgres-sql-eval
```

If you want to reset the Postgres server instance's state (e.g. memory leaks from transient connections), you can turn it off (and start it back up after):

```bash
docker stop postgres-sql-eval
# see that the container is still there:
docker container list -a
```

Some notes:

- You would need to stop other Postgres instances listening on port 5432 before running the above command.
- You only need to run the `docker create ...` once to create the image, and then subsequently only `docker start/stop postgres-sql-eval`.
- The data is persisted in `data/postgres`, so turning it off isn't critical. On the other hand, if you delete the `data/postgres` folder, then all is lost T.T
- While we will use Docker for deploying Postgres and the initialization, you are free to modify the scripts/instructions to work with your local installation.

### Import Data into Postgres

The data for importing is in the `defog-data` repository which we cloned earlier. Each folder contains the metadata and data corresponding to a single database (e.g. `academic` contains all the data required to reload the 'academic' database). We assume that you have a `psql` client installed locally. We will create a new database in our postgres instance for each of the 7 SQL databases with the following commands:

```bash
# set the following environment variables
cd defog-data # if you're not already in the defog-data directory
export DBPASSWORD="postgres"
export DBUSER="postgres"
export DBHOST="localhost"
export DBPORT=5432
./setup.sh
```

### Import Data into Snowflake

Should you wish to import the data into Snowflake, the setup instructions are also in the `defog-data` repository. After installing the [Snowflake CLI](https://docs.snowflake.com/en/user-guide/snowsql-install-config), configure your credentials as per the [docs](https://docs.snowflake.com/en/user-guide/snowsql-config) and set them as environment variables like below, then run the setup command.

```sh
export SFDBPASSWORD="your_password"
export SFDBUSER="your_username"
export SFDBACCOUNT="your_account"
export SFDBWAREHOUSE="your_warehouse"
./setup_snowflake.sh
```

Note that during evaluation you'll have to use the `_snowflake` question files in `/data`. The queries been modified to be valid on Snowflake databases.

### Import Data into BigQuery, MySQL, SQLite, SQL Server

The setup instructions for these database management systems are found in the `defog-data` repository. Configure your credentials accordingly, set up your environment variables, then translate and import the eval databases with the command:

```python
python translate_ddl_dialect.py
```

During evaluation, you'll have to set the right `--db_type` flag and use the corresponding `_{dialect}` question files in `/data`.

### Using Private Data (Optional)

If you have a private dataset that you do not want to make publicly available but would still like to repurpose the code here for evaluations, you can do so by following the steps below.

- Begin by creating a separate git repository for your private data, that has a `setup.py` file, similar to [defog-data](https://github.com/defog-ai/defog-data).
- Create the metadata and data files, and import them into your database. This is to allow our evaluation framework to run the generated queries with some actual data. You can refer to `defog-data`'s [metadata objects](https://github.com/defog-ai/defog-data/blob/main/defog_data/metadata.py) for the schema, and [setup.sh](https://github.com/defog-ai/defog-data/blob/main/setup.sh) as an example on how import the data into your database. We do not prescribe any specific folder structure, and leave it to you to decide how you want to organize your data, so long as you can import it into your database easily.
- To use our metadata pruning utilities, you would need to have the following defined:
  - A way to define joinable columns between tables. In our case, we call a dictionary [columns_join](https://github.com/defog-ai/defog-data/blob/db8c3d4c4004144d2b3ff5a2701529f5545f520f/defog_data/supplementary.py#L233) of database name to a nested dictionary of table tuples to column name tuples. You can refer to the raw data for an example of how we generate this dictionary.

Once all of the 3 above steps have completed, you would need to

- Install your data library as a dependency, by running `pip install -e .` (-e to automatically incorporate edits without reinstalling)
- Replace the associated function calls and variables in [prune_metadata_str](utils/pruning.py#L165) with your own imported functions and variables. Note that you might not name your package/module `defog_data_private.supplementary`, so do modify accordingly.

Some things to take note of:

- If you do not populate your database with data (ie only create the tables without inserting data), you would return empty dataframes most of the time (regardless of whether the query generated was what you want), and it would result in results matching all the time and generate a lot of false positives. Hence, you might want to consider populating your database with some meaningful data that would return different results if the queries should be different from what you want.
- If testing out on your private data, you would also need to change the questions file to point to your own questions file (tailored to your database schema).

### Runner

The runner calls is responsible for handling the configuration of work (e.g. parallelization / batching / model selected etc.) for each question/query pair.

We have provided a few common runners: `runners/openai_runner.py` for calling OpenAI's API (with parallelization support), `runners/anthropic_runner` for calling Anthropic's API, `runners/hf_runner.py` for calling a local Hugging Face model and finally, `runners/api_runner.py` makes it possible to use a custom API for evaluation.

## Running the Test

### OpenAI

Remember to have your API key (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`) set as an environment variable before running the test if you plan to call the OpenAI or Anthropic/other LLM API's accordingly.

To test it out with just 10 questions (instead of all 200), parallelized across 5 :

```bash
python main.py \
  -db postgres \
  -q "data/questions_gen_postgres.csv" "data/instruct_basic_postgres.csv" "data/instruct_advanced_postgres.csv" \
  -o results/openai_classic.csv results/openai_basic.csv results/openai_advanced.csv \
  -g oa \
  -f prompts/prompt_openai.json \
  -m o3-mini \
  -p 5 \
  -c 0
```

If testing with the latest `o1-*` models (which do not support system prompts), you should use a different prompt file, reduce parallel requests and increase the timeout:
```bash
python main.py \
  -db postgres \
  -q "data/questions_gen_postgres.csv" "data/instruct_basic_postgres.csv" "data/instruct_advanced_postgres.csv" \
  -o results/openai_o1mini_classic.csv results/openai_o1mini_basic.csv results/openai_o1mini_advanced.csv \
  -g oa \
  -f prompts/prompt_openai_o1.json \
  -m o1-mini \
  -p 1 \
  -t 120 \
  -c 0
```

### Anthropic

To test out the full suite of questions for claude-3:

```bash
python main.py \
  -db postgres \
  -q "data/questions_gen_postgres.csv" "data/instruct_basic_postgres.csv" "data/instruct_advanced_postgres.csv" \
  -o results/claude3_classic.csv results/claude3_basic.csv results/claude3_advanced.csv \
  -g anthropic \
  -f prompts/prompt_anthropic.md \
  -m claude-3-opus-20240229 \
  -p 5 \
  -c 0
```

### Hugging Face

To test it out with our fine-tuned sql model with just 10 questions (instead of all 200):

```bash
# use the -W option to ignore warnings about sequential use of transformers pipeline
python -W ignore main.py \
  -db postgres \
  -q "data/questions_gen_postgres.csv" "data/instruct_basic_postgres.csv" "data/instruct_advanced_postgres.csv" \
  -o results/hf_classic.csv results/hf_basic.csv results/hf_advanced.csv \
  -g hf \
  -f prompts/prompt.md \
  -m defog/llama-3-sqlcoder-8b \
  -c 0
```

We also support loading a peft adapter here as well via the `-a` flag. Note that the loading of the adapter with the model will take slightly longer than usual.

### vLLM

We also have a [vllm](https://blog.vllm.ai/) runner which uses the vLLM engine to run the inference altogether as a single batch. It is much faster to do so especially when `num_beams` > 1. You would have to pass in a single set of merged model weights, path to LoRA adapters if applicable, and the model architecture needs to be supported by vLLM. Here's a sample command:

```bash
python -W ignore main.py \
  -db postgres \
  -q "data/questions_gen_postgres.csv" "data/instruct_basic_postgres.csv" "data/instruct_advanced_postgres.csv" \
  -o results/vllm_classic.csv results/vllm_basic.csv results/vllm_advanced.csv \
  -g vllm \
  -f "prompts/prompt.md" \
  -m defog/llama-3-sqlcoder-8b \
  -a path/to_adapter \
  -c 0
```

Optionally, if you're running evals on a model that is quantized with AWQ, add the `-qz` or `--quantized` parameter. Only applicable for the vllm runner.

### Running with an API Server

If running with different settings, you can setup an api server to avoid reloading for each test setting and then run the tests subsequently. We enable setting up 2 types of api servers, namely the vllm api server, as well as the TGI server.

We also provide our custom modification of the vllm api server, which only returns the generated output.

#### VLLM API Server

```bash
# to set up a vllm server
python -m vllm.entrypoints.api_server \
    --model defog/defog-llama-3-sqlcoder-8b \
    --tensor-parallel-size 4 \
    --dtype float16

# to set up a vllm server that supports LoRA adapters
python -m vllm.entrypoints.api_server \
    --model defog/llama-3-sqlcoder-8b \
    --tensor-parallel-size 1 \
    --dtype float16 \
    --max-model-len 4096 \
    --enable-lora \
    --max-lora-rank 64

# to use our modified api server
python utils/api_server.py \
    --model defog/llama-3-sqlcoder-8b \
    --tensor-parallel-size 4 \
    --dtype float16 \
    --max-model-len 4096 \
    --enable-lora \
    --max-lora-rank 64

# to run sql-eval using the api runner - depending on how much your GPUs can take, can increase p and b to higher values
python main.py \
  -db postgres \
  -q "data/questions_gen_postgres.csv" \
  -o results/api.csv \
  -g api \
  -b 1 \
  -f prompts/prompt.md \
  --api_url "http://localhost:8000/generate" \
  --api_type "vllm" \
  -a path/to_adapter_if_applicable \
  -p 8
```

#### TGI API Server

You may consult the [TGI documentation](https://huggingface.co/docs/text-generation-inference/quicktour) for more information on how to set up a TGI server. Here's a sample command to set up a TGI server using a preset docker image and run the evaluation using the API runner. Note that you would want to change the number of shards and the model id accordingly, depending on how many gpu's you have available and your model of choice.

```bash
# to set up a tgi server
model="defog/llama-3-sqlcoder-8b"
docker run --gpus all \
  --shm-size 1g \
  -p 8000:80 \
  -v /models:/models ghcr.io/huggingface/text-generation-inference:2.0 \
  --model-id "${model}" \
  --max-best-of 4 \
  --max-input-tokens 3072 \
  --sharded true \
  --num-shard 4 \
  --hostname 0.0.0.0 \
  --port 80

# to run sql-eval using the api runner - depending on how much your GPUs can take, can increase p and b to higher values. Note that cuda graphs in tgi is optimized for batch sizes that are powers of 2 by default.
python main.py \
  -db postgres \
  -q "data/questions_gen_postgres.csv" \
  -o results/api.csv \
  -g api \
  -b 1 \
  -f prompts/prompt.md \
  --api_url "http://localhost:8000/generate" \
  --api_type "vllm" \
  -p 8
```

#### Multiple Prompts

If you'd like to test out a few prompts in a single run (to save the few minutes spent loading the model into GPU at the start of each run), you can specify a list of prompt files in `--prompt_file` (e.g. `-f prompts/prompt-1.md prompts/prompt-2.md prompts/prompt-3.md`), as well as a corresponding list of output files in `--output_file` (e.g. `-o results/results-1.csv results/results-2.csv results/results-3.csv`). The number of prompts and output files must be the same. Here's a sample command:

```bash
python -W ignore main.py \
  -db postgres \
  -q "data/questions_gen_postgres.csv" \
  -o results/results_1.csv results/results_2.csv \
  -g vllm \
  -f prompts/prompt_1.md prompts/prompt_2.md \
  -m defog/sqlcoder2
```

While you can do the same for the other runners, the time savings are most significant when loading a large model locally, vs calling an always-on API.

### Bedrock

```bash
python -W ignore main.py \
  -db postgres \
  -q "data/questions_gen_postgres.csv" \
  -o results/llama3_70b.csv \
  -g bedrock \
  -f prompts/prompt.md \
  -m meta.llama3-70b-instruct-v1:0
```

### Llama CPP

To run the eval using Llama CPP, you can use the following code. Before running this, you must install `llama-cpp-python` with the following (on Apple Silicon)

`CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python`

Note that llama-cpp-python library does not currently have beam search, and hence will have lower quality results.

```bash
python -W ignore main.py \
  -q "data/questions_gen_postgres.csv" \
  -db postgres \
  -o "results/llama_cpp.csv" \
  -g llama_cpp \
  -f "prompts/prompt.md" \
  -m path/to/model.gguf
```

### MLX

To run the eval using MLX, you can use the following code. Before running this, you must install `mlx-lm` package with `pip install mlx-lm`

Note that MLX does not currently have beam search, and hence will have lower quality results.

```bash
python -W ignore main.py \
  -db postgres \
  -q "data/questions_gen_postgres.csv" \
  -o "results/mlx_llama-3-sqlcoder-8b.csv" \
  -g mlx \
  -f "prompts/prompt.md" \
  -m mlx-community/defog-llama-3-sqlcoder-8b
```

### Gemini

Before running this, you need to set your credentials with `export GEMINI_API_KEY=<your_api_key>`. Then, install these packages with `pip install google-generative-ai`.

```bash
python main.py \
  -db postgres \
  -q "data/questions_gen_postgres.csv" "data/instruct_basic_postgres.csv" "data/instruct_advanced_postgres.csv" \
  -o "results/gemini_flash_basic.csv" "results/gemini_flash_basic.csv" "results/gemini_flash_advanced.csv" \
  -g gemini \
  -f "prompts/prompt_gemini.md" "prompts/prompt_gemini.md" "prompts/prompt_gemini.md" \
  -m gemini-2.0-flash-exp \
  -p 10
```

### Mistral

Before running this, you must create an account with [Mistral](https://mistral.ai/) and obtain an API key and store it with `export MISTRAL_API_KEY=<your_api_key>`. Then, install `mistralai` with `pip install mistralai`.

```bash
python -W ignore main.py \
  -db postgres \
  -q "data/questions_gen_postgres.csv" \
  -o "results/results.csv" \
  -g mistral \
  -f "prompts/prompt_mistral.md" \
  -m mistral-medium \
  -p 5 \
  -n 10
```

### Bedrock

Before running this, you would need to export the following environment variables for the boto3 client to work:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_DEFAULT_REGION`

```bash
python3 main.py \
  -db postgres \
  -q data/instruct_basic_postgres.csv data/instruct_advanced_postgres.csv data/questions_gen_postgres.csv \
  -o results/bedrock_llama_70b_basic.csv results/bedrock_llama_70b_advanced.csv results/bedrock_llama_70b_v1.csv \
  -g bedrock \
  -f prompts/prompt_cot_postgres.md \
  -m meta.llama3-70b-instruct-v1:0 \
  -c 0 \
  -p 10
```

### Deepseek

Before running this, you must create an account with [Deepseek](https://deepseek.com/) and obtain an API key and store it with `export DEEPSEEK_API_KEY=<your_api_key>`. Then, install `openai` with `pip install openai`. You can then run the following command:

#### Deepseek Chat
python main.py \
  -db postgres \
  -q "data/questions_gen_postgres.csv" "data/instruct_basic_postgres.csv" "data/instruct_advanced_postgres.csv" \
  -o results/deepseek_classic.csv results/deepseek_basic.csv results/deepseek_advanced.csv \
  -g deepseek \
  -f prompts/prompt_openai.json \
  -m deepseek-chat \
  -p 5 \
  -c 0

#### Deepseek Reasoner
python main.py \
  -db postgres \
  -q "data/questions_gen_postgres.csv" "data/instruct_basic_postgres.csv" "data/instruct_advanced_postgres.csv" \
  -o results/deepseek_classic.csv results/deepseek_basic.csv results/deepseek_advanced.csv \
  -g deepseek \
  -f prompts/prompt_openai_o1.json \
  -m deepseek-reasoner \
  -p 5 \
  -c 0


### Together

Before running this, you must create an account with [Together.ai](https://together.ai/) and obtain an API key and store it with `export TOGETHER_API_KEY=<your_api_key>`. Then, install `together` with `pip install together`. You can then run the following command:

```bash
python3 main.py \
  -db postgres \
  -q data/instruct_basic_postgres.csv data/instruct_advanced_postgres.csv data/questions_gen_postgres.csv \
  -o results/together_llama_70b_basic.csv results/together_llama_70b_advanced.csv results/together_llama_70b_v1.csv \
  -g together \
  -f prompts/prompt_together.json \
  -m "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo" \
  -c 0 \
  -p 10
```

## CLI Flags

You can use the following flags in the command line to change the configurations of your evaluation runs.

### Data-related parameters

| CLI Flags              | Description                                                                                                                                                                                                                                                                                          |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| -q, --questions_file   | CSV file that contains the test questions and true queries. If this is not set, it will default to the relevant `questions_gen_<db_type>.csv` file. It may be helpful to always end your questions*file name with `*<db_type>.csv` to ensure compatibility between the queries and selected db_type. |
| -n, --num_questions    | Use this to limit the total number of questions you want to test.                                                                                                                                                                                                                                    |
| -db, --db_type         | Database type to run your queries on. Currently supported types are `postgres` and `snowflake`.                                                                                                                                                                                                      |
| -d, --use_private_data | Use this to read from your own private data library.                                                                                                                                                                                                                                                 |
| -dp, --decimal_points  | Use this to specify the number of decimal points a result should be rounded to. This is `None` by default                                                                                                                                                                                            |

### Model-related parameters

| CLI Flags        | Description                                                                                                                                                                                                                                                                                                                                                                                       |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| -g, --model_type | Model type used. Make sure this matches the model used. Currently defined options in `main.py` are `oa` for OpenAI models, `anthropic` for Anthropic models, `hf` for Hugging Face models, `vllm` for a vllm runner, `api` for API endpoints, `llama_cpp` for llama cpp, `mlx` for mlx, `bedrock` for AWS bedrock API, `together` for together.ai's API           |
| -m, --model      | Model that will be tested and used to generate the queries. Some options for OpenAI models are chat models `gpt-4o` and `o3-mini`. Options for Anthropic include the latest claude-3 family of models (e.g. `claude-3-opus-20240229`). For Hugging Face, and VLLM models, simply use the path of your chosen model (e.g. `defog/sqlcoder`). |
| -a, --adapter    | Path to the relevant adapter model you're using. Only available for the `hf_runner`.                                                                                                                                                                                                                                                                                                              |
| --api_url        | The URL of the custom API you want to send the prompt to. Only used when model_type is `api`.                                                                                                                                                                                                                                                                                                     |
| -qz, --quantized | Indicate whether the model is an AWQ quantized model. Only available for `vllm_runner`.                                                                                                                                                                                                                                                                                                           |

### Inference-technique-related parameters

| CLI Flags              | Description  |     |
| ---------------------- |------------- | --- |
| -f, --prompt_file      | Markdown file with the prompt used for query generation. You can pass in a list of prompts to test sequentially without reloading the script. |
| -b, --num_beams        | Indicates the number of beams you want to use for beam search at inference. Only available for `hf_runner`, `vllm_runner`, and `api_runner`. |
| -c, --num_columns      | Number of columns, default 20. To not prune the columns, set it to 0. |
| -s, --shuffle_metadata | Shuffle metadata, default False. This shuffles the order of the tables within the schema and the order of the columns within each table but does not shift columns between tables (to preserve the structure of the database). |
| -k, --k_shot           | Used when you want to include k-shot examples in your prompt. Make sure that the column 'k_shot_prompt' exists in your questions_file. |
| --cot_table_alias      | (Experimental) Used when you want to include chain-of-thought instructions before the actual sql generation. Allowed values are `instruct`. If using `instruct`, make sure that the placeholder '{cot_instructions}' exists in your prompt file. `instruct` will get your model generate the chain-of-thought table aliases. |

### Execution-related parameters

| CLI Flags              | Description                                                                                                                                                                      |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| -o, --output_file      | Output CSV file that will store your results. You need to pass the same number of output file paths as the number of prompt files.                                               |
| -p, --parallel_threads | No. of parallel workers available for generating and processing queries                                                                                                          |
| -t, --timeout_gen      | No. of seconds before timeout occurs for query generation. The default is 30.0s.                                                                                                 |
| -u, --timeout_exec     | No. of seconds before timeout occurs for query execution on the database. The default is 10.0s.                                                                                  |
| -v, --verbose          | Prints details in command line.                                                                                                                                                  |
| --upload_url           | (optional) the URL that you want to report the results to. The server that serves this URL must have functionality that is similar to the sample server in `utils/webserver.py`. |
| --run_name             | (optional) the name of this run for logging purposes                                                                                                                             |

## Checking the Results

### Upload URL

If you would like to start a google cloud function to receive the results, you can use the `--upload_url` flag to specify the URL that you want to report the results to. Before running the evaluation code with this flag, you would need to create a server that serves at the provided URL. We have provided 2 sample cloud function endpoints for writing either to bigquery or postgres, in the `results_fn_bigquery` and `results_fn_postgres` folders. You may also implement your own server to take in similar arguments. Before deploying either cloud functions, you would need to set up the environment variables by making a copy of .env.yaml.template and renaming it to .env.yaml, and then filling in the relevant fields. For the bigquery cloud function, you would also need to put your service account's key.json file in the same folder, and put the file name in the `CREDENTIALS_PATH` field in the .env.yaml file.

After doing so, you can deploy the google cloud function:

```bash
# for uploading to bigquery
gcloud functions deploy results_bigquery \
  --source results_fn_bigquery \
  --entry-point bigquery \
  --env-vars-file results_fn_bigquery/.env.yaml \
  --runtime python311 \
  --memory 512MB \
  --trigger-http \
  --allow-unauthenticated \
  --gen2

# for uploading to postgres
gcloud functions deploy results_postgres \
  --source results_fn_postgres \
  --entry-point postgres \
  --env-vars-file results_fn_postgres/.env.yaml \
  --runtime python311 \
  --memory 512MB \
  --trigger-http \
  --allow-unauthenticated \
  --gen2
```

The cloud function's name is whatever comes after `gcloud functions deploy` (in this case, `results_bigquery`), and you can use it to check the logs of the function by running `gcloud functions logs read results_bigquery`.

You can then run the evaluation code with the `--upload_url` flag to report the results to the cloud function. The cloud function will then write the results to the relevant database.

```bash
python main.py \
  -db postgres \
  -o results/test.csv \
  -g oa \
  -f prompts/prompt_openai.json \
  -m gpt-4o-mini \
  -n 1 \
  --upload_url <your cloud function url>
```

If you would like to always report your results to an upload_url, even if it's not explicitly provided, you can set it in your environment variables as `SQL_EVAL_UPLOAD_URL`

#### Testing the function locally

If you'd like to modify the functions and test it out locally, you can run these sample commands to deploy the function locally and then trigger the openai runner:

```bash
functions-framework --target bigquery --source results_fn_bigquery --debug
python main.py \
  -db postgres \
  -o results/test.csv \
  -g oa \
  -f prompts/prompt_openai.json \
  -m gpt-4o-mini \
  -n 1 \
  --upload_url http://127.0.0.1:8080/
```

## Misc

We welcome contributions to our project, specifically:

- Dataset
  - Adding new database schema/data
- Framework code
  - Improving existing generators/runners (e.g. adding new metrics)

Please see [CONTRIBUTING.md](https://github.com/defog-ai/sql-generation-evaluation/blob/main/CONTRIBUTING.md) for more information.
