# SQL Generation Evaluation

This repository contains the code that Defog uses for sql generation evaluation. It is based off the [spider](https://github.com/taoyds/spider) datasets' schema, but with a new set of hand-selected questions and queries grouped by query category.

## Introduction

Our testing procedure comprises the following steps. For each question/query pair:
1. We generate a query (could be from a LLM).
2. We run both the "gold" query and the generated query on their respective postgres database and obtain 2 dataframes with the results.
3. We compare the 2 dataframes using an "exact" and a "subset" match. TODO add link to blogpost.
4. We log these alongside other metrics of interest (eg tokens used, latency) and aggregate the results for reporting.

## Getting Started

This is a comprehensive set of instructions that assumes basic familiarity with the command line, docker, running SQL queries on a database, and common python data manipulation libraries involved (`pandas`).

### Start Postgres Instance

Firstly, you would need to setup the databases used to run the queries on. We use postgres here, since it is the most common OSS database with the widest distribution and usage in production. In addition, we would recommend using docker to do this, as it is the easiest way to get started. You can install docker [here](https://docs.docker.com/get-docker/). Once you have docker installed, you can create the docker container, and then start the postgres database using the following commands. We recommend mounting a volume on `data/postgres` to persist the data, as well as `data/export` to make it easier to import the data. To create the container, run:

```bash
mkdir data/postgres data/export
docker create --name postgres-sql-eval -e POSTGRES_PASSWORD=postgres -p 5432:5432 -v $(pwd)/data/postgres:/var/lib/postgresql/data -v $(pwd)/data/export:/export postgres:14-alpine
```

To start the container, run:
```bash
docker start postgres-sql-eval
```

If you want to reset the postgres server instance's state (eg memory leaks from transient connections), you can turn it off (and start it back up after):
```bash
docker stop postgres-sql-eval
# see that the container is still there:
docker container list -a
```

Some notes:
- You would need to stop other postgres instances listening on port 5432 before running the above command.
- You only need to run the `docker create ...` once to create the image, and then subsequently only `docker start/stop postgres-sql-eval`. 
- The data is persisted in `data/postgres`, so turning it off isn't critical. On the other hand, if you delete the `data/postgres` folder, then all is lost T.T
- While we will use docker for deploying postgres and the initialization, you are free to modify the scripts/instructions to work with your local installation.


### Import data into Postgres

The data for importing is already in the exported sql dumps in the `data/export` folder. Each sql file corresponds to its own database (eg `data/export/academic.sql` contains all the data required to reload the academic database). We will create a new database for each database, in `postgres-sql-eval`.

```bash
./data/init_db.sh
```

## Misc

We welcome contributions to our project. Please see [CONTRIBUTING.md](https://github.com/defog-ai/sql-generation-evaluation/blob/main/CONTRIBUTING.md) for more information.