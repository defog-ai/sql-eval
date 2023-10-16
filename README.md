# SQL Generation Evaluation

[![tests](https://github.com/defog-ai/sql-generation-evaluation/actions/workflows/main.yml/badge.svg)](https://github.com/defog-ai/sql-generation-evaluation/actions/workflows/main.yml)

This repository contains the code that Defog uses for the evaluation of generated SQL. It's based off the schema from the [Spider](https://github.com/taoyds/spider), but with a new set of hand-selected questions and queries grouped by query category. For an in-depth look into our process of creating this evaluation approach, see [this](https://defog.ai/blog/open-sourcing-sqleval/).

## Introduction

Our testing procedure comprises the following steps. For each question/query pair:
1. We generate a SQL query (possibly from an LLM).
2. We run both the "gold" query and the generated query on their respective Postgres database to obtain 2 dataframes with the results.
3. We compare the 2 dataframes using an "exact" and a "subset" match. TODO add link to blogpost.
4. We log these alongside other metrics of interest (e.g. tokens used, latency) and aggregate the results for reporting.

## Getting Started

This is a comprehensive set of instructions that assumes basic familiarity with the command line, Docker, running SQL queries on a database, and common Python data manipulation libraries (e.g. pandas).

### Install Dependencies

Firstly, install all Python libraries listed in the `requirements.txt` file. You would also need to download the spacy model used in the NER heuristic for our [metadata-pruning method](https://github.com/defog-ai/sql-eval/blob/main/utils/pruning.py). Also, you would need to clone the repository where we store our database data and schema, and install the library.
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
git clone https://github.com/defog-ai/defog-data.git
cd defog-data
pip install -e .
```

### Start Postgres Instance

Next, you would need to set up the databases that the queries are executed on. We use Postgres here, since it is the most common OSS database with the widest distribution and usage in production. In addition, we would recommend using Docker to do this, as it is the easiest way to get started. You can install Docker [here](https://docs.docker.com/get-docker/). 

Once you have Docker installed, you can create the Docker container and start the Postgres database using the following commands. We recommend mounting a volume on `data/postgres` to persist the data, as well as `data/export` to make it easier to import the data. To create the container, run:

```bash
mkdir data/postgres data/export
docker create --name postgres-sql-eval -e POSTGRES_PASSWORD=postgres -p 5432:5432 -v $(pwd)/data/postgres:/var/lib/postgresql/data -v $(pwd)/data/export:/export postgres:14-alpine
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

### Using Private Data (Optional)

If you have a private dataset that you do not want to make publicly available but would still like to repurpose the code here for evaluations, you can do so by following the steps below.
- Begin by creating a separate git repository for your private data, that has a `setup.py` file, similar to [defog-data](https://github.com/defog-ai/defog-data).
- Create the metadata and data files, and import them into your database. This is to allow our evaluation framework to run the generated queries with some actual data. You can refer to `defog-data`'s [metadata objects](https://github.com/defog-ai/defog-data/blob/main/defog_data/metadata.py) for the schema, and [setup.sh](https://github.com/defog-ai/defog-data/blob/main/setup.sh) as an example on how import the data into your database. We do not prescribe any specific folder structure, and leave it to you to decide how you want to organize your data, so long as you can import it into your database easily.
- To use our metadata pruning utilities, you would need to have the following defined:
  - A way to load your embeddings. In our case, we call a function [load_embeddings](utils/pruning.py#L173) from `defog-data`'s supplementary module to load a dictionary of database name to a tuple of the 2D embedding matrix (num examples x embedding dimension) and the associated text metadata for each row/example. If you would like to see how we generate this tuple, you may refer to [generate_embeddings](https://github.com/defog-ai/defog-data/blob/main/defog_data/supplementary.py#L11) in the `defog-data` repository.
  - A way to load columns associated with various named entities. In our case, we call a dictionary [columns_ner](utils/pruning.py#L178) of database name to a nested dictionary that maps each named entity type to a list of column metadata strings that are associated with that named entity type. You can refer to the raw data [here](https://github.com/defog-ai/defog-data/blob/main/defog_data/supplementary.py#L77) for an example of how we generate this dictionary.
  - A way to define joinable columns between tables. In our case, we call a dictionary [columns_join](utils/pruning.py#L179) of database name to a nested dictionary of table tuples to column name tuples. You can refer to the raw data [here](https://github.com/defog-ai/defog-data/blob/main/defog_data/supplementary.py#L207) for an example of how we generate this dictionary.

Once all of the 3 above steps have completed, you would need to
- Install your data library as a dependency, by running `pip install -e .` (-e to automatically incorporate edits without reinstalling)
- Replace the associated function calls and variables in [prune_metadata_str](utils/pruning.py#L165) with your own imported functions and variables. Note that you might not name your package/module `defog_data_private.supplementary`, so do modify accordingly.

Some things to take note of:
- If you do not populate your database with data (ie only create the tables without inserting data), you would return empty dataframes most of the time (regardless of whether the query generated was what you want), and it would result in results matching all the time and generate a lot of false positives. Hence, you might want to consider populating your database with some meaningful data that would return different results if the queries should be different from what you want.
- If testing out on your private data, you would also need to change the questions file to point to your own questions file (tailored to your database schema).

### Query Generator

To test your own query generator with our framework, you would need to extend [Query Generator](query_generators/query_generator.py) and implement the [generate_query](query_generators/query_generator.py#L18) method to return the query of interest. We create a new class for each question/query pair to isolate each pair's runtime state against the others when running concurrently. You can also reference [OpenAIQueryGenerator](query_generators/openai.py) which implements `Query Generator` and uses a simple prompt to send a message over to OpenAI's API. Feel free to extend it for your own use.

If there are functions that are generally useful for all query generators, they can be placed in the `utils` folder. If you need to incorporate specific verbose templates (e.g. for prompt testing), you can store them in the `prompts` folder, and later import them. Being able to version control the prompts in a central place has been a productivity win for our team.

### Runner

Having implemented the query generator, the next piece of abstraction would be the runner. The runner calls the query generator, and is responsible for handling the configuration of work (e.g. parallelization / batching / model selected etc.) to the query generator for each question/query pair. We have provided 2 most common runners: `eval/openai_runner.py` for calling OpenAI's API (with parallelization support) and `eval/hf_runner.py` for calling a local Hugging Face model. When testing your own query generator with an existing runner, you can replace the `qg_class` in the runner's code with your own query generator class.

## Running the Test

### OpenAI
Remember to have your OpenAI API key (`OPENAI_API_KEY="sk-..."`) set as an environment variable before running the test if you plan to call the OpenAI API (or Anthropic/other LLM API's accordingly).

To test it out with just 10 questions (instead of all 175), parallelized across 5 :

```bash
mkdir results # create directory for storing results
python main.py \
  -q data/questions_gen.csv \
  -o results/my_query_generator.csv \
  -g oa \
  -f prompts/prompt.md \
  -m gpt-3.5-turbo-0613 \
  -n 10 \
  -p 5
```

### Hugging Face
To test it out with our fine-tuned sql model with just 10 questions (instead of all 175):

```bash
mkdir results #create directory for storing results

# use the -W option to ignore warnings about sequential use of transformers pipeline
python -W ignore main.py \
  -q data/questions_gen.csv \
  -o results/results.csv \
  -g hf \
  -f prompts/prompt.md \
  -m defog/starcoder-finetune-v3 \
  -n 10
```


### CLI Flags
You can use the following flags in the command line to change the configurations of your evaluation runs.
| CLI Flags     | Description |
|-------------|-------|
|  -q, --questions_file   |  CSV file that contains the test questions and true queries.   |
| -n, --num_questions  |  Use this to limit the total number of questions you want to test.  |
|  -g, --model_type   |  Model type used. Make sure this matches the model used. Currently defined options in `main.py` are `oa` for OpenAI models and `hf` for Hugging Face models.   |
|  -m, --model   |  Model that will be tested and used to generate the queries. Currently defined options for OpenAI models are chat models `gpt-3.5-turbo-0613` and `gpt-4-0613`, and non-chat model `text-davinci-003`. For Hugging Face models, simply use the path of your chosen model (e.g. `defog/sqlcoder`).  |
|  -f, --prompt_file   |  Markdown file with the prompt used for query generation.  |
|  -d, --use_defog_data  |  Use this to toggle between using the public data or your own private data.  |
|  -o, --output_file   |  Output CSV file that will store your results.   |
| -p, --parallel_threads  |  The default no. of parallel threads is 5. Decrease this to 1 for gpt-4 to avoid the rate limit error. Parallelization support is currently only defined for OpenAI models.  |
| -t, --timeout_gen  |  No. of seconds before timeout occurs for query generation. The default is 30.0s. |
| -u, --timeout_exec  |  No. of seconds before timeout occurs for query execution on the database. The default is 10.0s.  |
| -v, --verbose  |  Prints details in command line. |

## Checking the Results
To better understand your query generator's performance, you can explore the results generated and aggregated for the various metrics that you care about. Happy iterating!

## Misc

We welcome contributions to our project, specifically:
- Dataset
  - Adding new database schema/data
- Framework code
  - New query generators/runners (in the [query_generators](query_generators) and [eval](eval) folders respectively)
  - Improving existing generators/runners (e.g. adding new metrics)

Please see [CONTRIBUTING.md](https://github.com/defog-ai/sql-generation-evaluation/blob/main/CONTRIBUTING.md) for more information.