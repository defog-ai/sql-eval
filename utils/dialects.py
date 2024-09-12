import sqlglot
from sqlglot import parse_one, exp
from tqdm import tqdm
import re

try:
    from google.cloud import bigquery
except:
    pass
import os
import time
import concurrent.futures
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
import asyncio
import json
import pandas as pd
import logging
import sqlite3

# Suppress all logs from sqlglot
logging.getLogger("sqlglot").setLevel(logging.CRITICAL)
idk_list = list(pd.read_csv("data/idk.csv")["query"].unique())

######## GENERIC FUNCTIONS ########


def get_sql_tables(sql, db_type):
    """
    Get all tables in the sql query using sqlglot
    """
    table_list = set()
    try:
        for table in parse_one(sql, dialect=db_type).find_all(exp.Table):
            if len(table.name) > 1:
                table_list.add(table.name)
    except Exception as e:
        print(sql)
        print("Error parsing sql", e)
    return table_list


def get_all_tables_md(table_metadata_string):
    """
    Get all tables in the table metadata string. This is only used for BigQuery.
    """
    all_tables = set()
    for table in table_metadata_string.split("CREATE TABLE"):
        if "(" in table:
            table_name = table.split("(")[0].strip()
            all_tables.add(table_name.lower())
    # remove schema names
    all_tables = set([table.split(".")[-1] for table in all_tables])
    return all_tables


async def amend_invalid_sql(
    model: str,
    question: str,
    sql_list: list,
    err_msg_list: list,
    db_ddl: str,
    instructions: str,
    db_type: str = "postgres",
) -> str:
    """
    Use LLM to correct a list of invalid SQL queries given a common question, specific error message and the database schema
    """
    openai = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "ADD_YOUR_OPENAI_KEY_HERE")
    )
    completion_dict_list = []
    for i in range(len(sql_list)):
        sql = sql_list[i]
        err_msg = err_msg_list[i]
        if err_msg == "":
            completion_dict_list.append(
                {
                    "reason": "Already valid SQL",
                    "sql": sql,
                }
            )
            continue
        else:
            messages = [
                {
                    "role": "system",
                    "content": f"""Your task is to correct an invalid SQL query given a question, instructions and the database schema.""",
                },
                {
                    "role": "user",
                    "content": f"""The query would run on a `{db_type}` database whose schema is represented in this string:
        {db_ddl}

        Instructions for the task:
        {instructions}

        Question: {question}
        Invalid SQL: {sql}
        Error message: {err_msg}

        Format your response as a valid JSON string with reason and sql keys. 
        Your response should look like the string below:
        {{ "reason": "Your reasoning for the response",
            "sql": "The corrected SQL query that is valid on {db_type}",
        }}

        Do not include any other information before and after the JSON string.
        """,
                },
            ]

            completion = await openai.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=2000,
                temperature=0,
                # top_p=0.5,
                response_format={"type": "json_object"},
            )
            completion = completion.choices[0].message.content
            try:
                completion_dict = json.loads(completion)
            except:
                print(f"Error parsing completion {completion}. Retrying...", flush=True)
                # retry
                return await amend_invalid_sql(
                    model=model,
                    question=question,
                    sql_list=sql_list,
                    err_msg_list=err_msg_list,
                    db_ddl=db_ddl,
                    instructions=instructions,
                    db_type=db_type,
                )
            completion_dict_list.append(completion_dict)
    return completion_dict_list


async def amend_invalid_sql_concurr(df, model, max_concurrent, dialect):
    """
    Run amend_invalid_sql concurrently on a DataFrame
    """

    async def _process_row(index, row):
        result = await amend_invalid_sql(
            model=model,
            question=row["question"],
            sql_list=row[f"sql_{dialect}_list"],
            err_msg_list=row["err_msg_list"],
            db_ddl=row[f"table_metadata_{dialect}"],
            instructions=row.get("instructions", ""),
            db_type=dialect,
        )
        return index, result

    tasks = []
    semaphore = asyncio.Semaphore(max_concurrent)

    async def sem_task(index, row):
        async with semaphore:
            return await _process_row(index, row)

    tasks = [sem_task(index, row) for index, row in df.iterrows()]
    results = [await f for f in tqdm_asyncio.as_completed(tasks, total=len(tasks))]

    # Order results according to the order they were processed
    results.sort(key=lambda x: x[0])
    ordered_results = [result[1] for result in results]

    return ordered_results


def remove_unparseable_sql(sql):
    """
    Remove unparseable elements from the sql
    """
    if " ~* '" in sql:
        sql = sql.replace(" ~* '", " ILIKE '")
    if " ~ '" in sql:
        sql = sql.replace(" ~ '", " LIKE '")
    return sql


def sql_to_dialect(sql, db_type, dialect):
    """
    Translates sql of db_type to another dialect with sqlglot.
    Does not have any post-processing.
    """
    sql = remove_unparseable_sql(sql)
    translated = sqlglot.transpile(sql, read=db_type, write=dialect)
    translated = translated[0]
    return translated


def ddl_to_dialect(ddl, db_type, dialect):
    """
    Preprocesses the ddl and then translates it to another dialect with sqlglot.
    """

    ddl = ddl.replace("UNIQUE", "")
    ddl = ddl.replace(" PRIMARY KEY", "")
    try:
        translated = sqlglot.transpile(ddl, read=db_type, write=dialect, pretty=True)
    except Exception as e:
        print("Error transpiling ddl", e)
        print(ddl)
        raise
    translated = "\n".join(translated)
    return translated


def get_schema_names(ddl):
    """
    Get all schema names from the ddl.
    """
    schema_names = set()
    table_names = re.findall(r"CREATE TABLE ([a-zA-Z_][a-zA-Z0-9_\.]*) \(", ddl)
    for table_name in table_names:
        if "." in table_name:
            schema_name = table_name.split(".")[0]
            schema_names.add(schema_name)
    return schema_names


def ddl_remove_schema(translated_ddl):
    """
    Remove schema names from the translated ddl.
    """
    schema_names = get_schema_names(translated_ddl)
    for schema_name in schema_names:
        translated_ddl = translated_ddl.replace(
            f"CREATE TABLE {schema_name}.", "CREATE TABLE "
        )
        translated_ddl = translated_ddl.replace(
            f"CREATE SCHEMA IF NOT EXISTS {schema_name}", ""
        )
    return translated_ddl


def sql_remove_schema(translated_sql, table_metadata_string):
    """
    Remove schema names from the translated sql.
    """
    schema_names = get_schema_names(table_metadata_string)
    for schema_name in schema_names:
        translated_sql = translated_sql.replace(f"{schema_name}.", "")
    return translated_sql


######## BIGQUERY FUNCTIONS ########


def sql_to_bigquery(sql, db_type, table_metadata_string, db_name, row_idx):
    """
    Translates sql of db_type to bigquery dialect.
    Returns translated sql and translated test sql for testing
    """
    translated = sql_to_dialect(sql, db_type, "bigquery")

    # Replace all tables in the sql with db_name.table
    sql_tables = get_sql_tables(sql, db_type)
    all_tables = get_all_tables_md(table_metadata_string)
    normalized_sql_tables = set([table.lower() for table in sql_tables])
    table_list = normalized_sql_tables.intersection(all_tables)
    # revert to original case
    final_table_list = set(
        [table for table in sql_tables if table.lower() in table_list]
    )
    if (final_table_list == set()) and (sql + ";" not in idk_list):
        print("No tables found in sql. Skipping...")
        print(sql)
        return None, None

    # remove schema names if any
    translated = sql_remove_schema(translated, table_metadata_string)
    translated_test = translated
    for table in final_table_list:
        translated = re.sub(
            rf"\b {table}\b(?![\.])", rf" {db_name}.{table}", translated
        )
        translated_test = re.sub(
            rf"\b {table}\b(?![\.])",
            rf" test{row_idx}_{db_name}.{table}",
            translated_test,
        )

    return translated, translated_test


def ddl_to_bigquery(ddl, db_type, db_name, row_idx):
    """
    Translates ddl of db_type to bigquery dialect.
    BigQuery requires tables to be prefixed with the dataset name.
    This function adds the dataset name to the table names in the ddl.
    Returns translated ddl and translated test ddl for testing.
    """

    translated = ddl_to_dialect(ddl, db_type, "bigquery")
    translated = ddl_remove_schema(translated)

    # if any of reserved keywords in the non-comments section of ddl, enclose them with backticks
    reserved_keywords = ["long"]
    segments = re.split(r"(/\*.*?\*/)", translated)
    for i, segment in enumerate(segments):
        if not segment.startswith("/*"):
            for keyword in reserved_keywords:
                segment = re.sub(
                    rf'(?<!")\b{keyword}\b(?!")',
                    f"`{keyword}`",
                    segment,
                    flags=re.IGNORECASE,
                )
            segments[i] = segment
    translated = "".join(segments)

    translated = translated.replace(")\nCREATE", ");\nCREATE")
    translated = re.sub(r"SERIAL(PRIMARY KEY)?", "INT64", translated)
    translated = re.sub(
        r"NOT NULL DEFAULT CURRENT_TIMESTAMP\(\)",
        "DEFAULT CAST(CURRENT_TIMESTAMP() AS DATETIME) NOT NULL",
        translated,
    )
    translated += ";"
    translated_ddl = translated.replace("CREATE TABLE ", f"CREATE TABLE {db_name}.")
    translated_ddl_test = translated.replace(
        "CREATE TABLE ", f"CREATE TABLE test{row_idx}_{db_name}."
    )
    return translated_ddl, translated_ddl_test


def create_bq_db(client, creds, db_name, table_metadata_string_test, row_idx):
    """
    Create a test BigQuery dataset and tables from the table_metadata_string
    """

    db_name = f"test{row_idx}_" + db_name
    # Create a test dataset
    dataset_id = f"{creds['project']}.{db_name}"
    dataset = bigquery.Dataset(dataset_id)
    dataset.location = "US"

    try:
        created_dataset = client.create_dataset(dataset, timeout=30, exists_ok=True)
        # print("Dataset created or already exists. Full ID:", created_dataset.full_dataset_id)
    except Exception as e:
        print(f"Error creating dataset `{dataset_id}`: {e}")
        raise

    try:
        client.query(table_metadata_string_test)
        # print(f"Tables for `{db_name}` created successfully")
    except Exception as e:
        print(f"Error creating tables for `{dataset_id}`: {e}")
        raise


def delete_bq_db(client, creds, db_name, row_idx):
    """
    Delete a test BigQuery dataset
    """

    db_name = f"test{row_idx}_" + db_name

    # Delete the test dataset
    dataset_id = f"{creds['project']}.{db_name}"
    try:
        client.delete_dataset(dataset_id, delete_contents=True, not_found_ok=True)
        # print(f"Dataset `{db_name}` deleted successfully")
    except Exception as e:
        print(f"Error deleting dataset. Pls delete `{dataset_id}` manually: {e}")


def test_valid_md_bq(
    creds, sql_test_list, db_name, table_metadata_string_test, row_idx
):
    """
    Test the validity of the metadata and list of sql in BigQuery.
    This will create a test dataset and tables, run the test sqls and delete the test dataset.
    """
    from google.cloud import bigquery

    validity_tuple_list = []
    test_db = f"test{row_idx}_{db_name}"
    client = bigquery.Client(project=creds["project"])
    # create a test db
    try:
        create_bq_db(client, creds, db_name, table_metadata_string_test, row_idx)
    except Exception as e:
        delete_bq_db(client, creds, db_name, row_idx)
        error_tuple = (False, "Error creating test db: " + str(e))
        validity_tuple_list.extend([error_tuple] * len(sql_test_list))
        return validity_tuple_list
    time.sleep(2)

    # run the sqls
    for sql_test in sql_test_list:
        tries = 0
        error_msg = ""
        validity_added = False
        while tries < 5 and not validity_added:
            try:
                query_job = client.query(sql_test)
                results = query_job.result()
                validity_tuple_list.append((True, ""))
                validity_added = True
            except Exception as e:
                error_msg = str(e)
                if any(
                    x in error_msg for x in ["Not found: Table", "Not found: Dataset"]
                ):
                    # print(f"Retrying...{e}")
                    tries += 1
                    time.sleep(4)
                else:
                    validity_tuple_list.append((False, error_msg))
                    validity_added = True
        if not validity_added:
            validity_tuple_list.append((False, error_msg))

    delete_bq_db(client, creds, db_name, row_idx)
    return validity_tuple_list


def test_valid_md_bq_concurr(df, db_creds_all, sql_list_col, table_metadata_col):
    """
    Run test_valid_md_bq concurrently on a DataFrame
    """

    futures_to_index = {}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Prepare futures for each row in the DataFrame
        futures = [
            executor.submit(
                test_valid_md_bq,
                db_creds_all["bigquery"],
                row[sql_list_col],
                row["db_name"],
                row[table_metadata_col],
                row.get("index", str(index)),
            )
            for index, row in df.iterrows()
        ]

        # Map each future to its index
        for index, future in enumerate(futures):
            futures_to_index[future] = index

        # Collect results as they complete and map them back to their indices
        results = [None] * len(futures)
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            index = futures_to_index[future]
            results[index] = future.result()

    return results


def test_valid_bq(creds, sql_test_list, db_name):
    """
    Test the validity of the sql on an existing DB in BigQuery.
    If the sql is valid and returns non-empty results, return True.
    """
    from google.cloud import bigquery

    validity_tuple_list = []
    client = bigquery.Client(project=creds["project"])
    # run the sqls
    for sql_test in sql_test_list:
        error_msg = ""
        try:
            query_job = client.query(sql_test)
            results = query_job.result()
            # check if results empty
            if results:
                validity_tuple_list.append((True, ""))
            else:
                validity_tuple_list.append(
                    (
                        False,
                        "Results are empty, which means SQL is wrong and does not answer the question. Or data values in defog-data databases could be outdated.",
                    )
                )
        except Exception as e:
            error_msg = str(e)
            validity_tuple_list.append((False, error_msg))

    return validity_tuple_list


######## MYSQL FUNCTIONS ########


def sql_to_mysql(sql, db_type, table_metadata_string):
    """
    Translates sql of db_type to MySQL dialect.
    For MySQL, there's no need to prefix tables with db_name so translated and translated test sql are the same.
    Returns translated sql and translated test sql for testing.
    """
    translated = sql_to_dialect(sql, db_type, "mysql")
    translated = sql_remove_schema(translated, table_metadata_string)
    translated_test = translated
    return translated, translated_test


def ddl_to_mysql(ddl, db_type, db_name, row_idx):
    """
    Translates ddl of db_type to MySQL dialect.
    Returns translated ddl and translated test ddl for testing.
    """
    translated = ddl_to_dialect(ddl, db_type, "mysql")
    translated = ddl_remove_schema(translated)

    # if any of reserved keywords in the non-comments section of ddl, enclose them with backticks
    reserved_keywords = ["long"]
    segments = re.split(r"(/\*.*?\*/)", translated)
    for i, segment in enumerate(segments):
        if not segment.startswith("/*"):
            for keyword in reserved_keywords:
                segment = re.sub(
                    rf'(?<!")\b{keyword}\b(?!")',
                    f"`{keyword}`",
                    segment,
                    flags=re.IGNORECASE,
                )
            segments[i] = segment

    translated = "".join(segments)

    translated = translated.replace(")\nCREATE", ");\nCREATE")
    translated = re.sub(r"VARCHAR(?!\()", "VARCHAR(255)", translated)
    translated += ";"
    translated_ddl = translated.replace("CREATE TABLE ", f"CREATE TABLE {db_name}.")
    translated_ddl_test = translated.replace(
        "CREATE TABLE ", f"CREATE TABLE test{row_idx}_{db_name}."
    )
    return translated_ddl, translated_ddl_test


def create_mysql_db(creds, db_name, table_metadata_string_test, row_idx):
    """
    Create a test MySQL database and tables from the table_metadata_string
    """
    import mysql.connector
    from mysql.connector import errorcode

    test_db_name = f"test{row_idx}_" + db_name
    try:
        conn = mysql.connector.connect(**creds)
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE {test_db_name};")
        # print(f"Database `{test_db_name}` created successfully")
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print(f"Database `{test_db_name}` does not exist")
        elif "Lost connection" in str(err):
            time.sleep(2)
            print(f"Lost connection for `{test_db_name}`. Retrying...")
            create_mysql_db(creds, db_name, table_metadata_string_test, row_idx)
        else:
            print(err)
        raise

    try:
        cursor.execute(table_metadata_string_test)
        # print(f"Tables for `{test_db_name}` created successfully")
    except Exception as e:
        print(f"Error creating tables for `{test_db_name}`: {e}")
        print(table_metadata_string_test)
        raise
    finally:
        if "cursor" in locals():
            cursor.close()
        if "conn" in locals():
            conn.close()


def delete_mysql_db(creds, db_name, row_idx):
    """
    Delete a test MySQL database
    """
    import mysql.connector
    from mysql.connector import errorcode

    test_db_name = f"test{row_idx}_" + db_name

    # Delete the test database
    try:
        conn = mysql.connector.connect(**creds)
        cursor = conn.cursor()
        cursor.execute(f"DROP DATABASE {test_db_name};")
        # print("Database deleted successfully")
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)
    finally:
        if "cursor" in locals():
            cursor.close()
        if "conn" in locals():
            conn.close()


def test_valid_md_mysql(
    creds, sql_test_list, db_name, table_metadata_string_test, row_idx
):
    """
    Test the validity of the metadata and sql in MySQL.
    This will create a test dataset and tables, run the sql and delete the test dataset.
    """
    import mysql.connector

    validity_tuple_list = []
    test_db = f"test{row_idx}_{db_name}"
    # create a test db
    try:
        create_mysql_db(creds, db_name, table_metadata_string_test, row_idx)
    except Exception as e:
        if "cursor" in locals():
            cursor.close()
        if "conn" in locals():
            conn.close()
        delete_mysql_db(creds, db_name, row_idx)
        error_tuple = (False, "Error creating test db: " + str(e))
        validity_tuple_list.extend([error_tuple] * len(sql_test_list))
        return validity_tuple_list

    # run the sqls
    for sql_test in sql_test_list:
        tries = 0
        error_msg = ""
        validity_added = False
        while tries < 3 and not validity_added:
            try:
                conn = mysql.connector.connect(**creds)
                cursor = conn.cursor()

                use_db = f"USE {test_db};"
                cursor.execute(use_db)
                cursor.execute(sql_test)
                results = cursor.fetchall()
                cursor.close()
                conn.close()
                validity_tuple_list.append((True, ""))
                validity_added = True
            except Exception as e:
                error_msg = str(e)
                if (
                    "doesn't exist" in error_msg and "Table" in error_msg
                ) or "Lost connection" in error_msg:
                    tries += 1
                    time.sleep(2)
                else:
                    # print("Error running sql:", e)
                    validity_tuple_list.append((False, error_msg))
                    validity_added = True
        if not validity_added:
            validity_tuple_list.append((False, error_msg))

    if "cursor" in locals():
        cursor.close()
    if "conn" in locals():
        conn.close()
    delete_mysql_db(creds, db_name, row_idx)
    return validity_tuple_list


def test_valid_md_mysql_concurr(df, db_creds_all, sql_list_col, table_metadata_col):
    """
    Run test_valid_md_mysql concurrently on a DataFrame
    """
    futures_to_index = {}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Prepare futures for each row in the DataFrame
        futures = [
            executor.submit(
                test_valid_md_mysql,
                db_creds_all["mysql"],
                row[sql_list_col],
                row["db_name"],
                row[table_metadata_col],
                row.get("index", str(index)),
            )
            for index, row in df.iterrows()
        ]

        # Map each future to its index
        for index, future in enumerate(futures):
            futures_to_index[future] = index

        # Collect results as they complete and map them back to their indices
        results = [None] * len(futures)
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            index = futures_to_index[future]
            results[index] = future.result()

    return results


def test_valid_mysql(creds, sql_test_list, db_name):
    """
    Test the validity of the sql on an existing DB in MySQL.
    If the sql is valid and returns non-empty results, return True.
    """
    import mysql.connector

    validity_tuple_list = []
    # run the sqls
    for sql_test in sql_test_list:
        error_msg = ""
        try:
            conn = mysql.connector.connect(**creds)
            cursor = conn.cursor()
            use_db = f"USE {db_name};"
            cursor.execute(use_db)
            cursor.execute(sql_test)
            results = cursor.fetchall()
            # check if results empty
            if results:
                validity_tuple_list.append((True, ""))
            else:
                validity_tuple_list.append(
                    (
                        False,
                        "Results are empty, which means SQL is wrong and does not answer the question. Or data values in defog-data databases could be outdated.",
                    )
                )
        except Exception as e:
            error_msg = str(e)
            validity_tuple_list.append((False, error_msg))

    if "cursor" in locals():
        cursor.close()
    if "conn" in locals():
        conn.close()
    return validity_tuple_list


def instructions_to_mysql(instructions):
    """
    Convert the PostgreSQL-specific instructions to MySQL dialect.
    """

    def format_date_for_mysql(interval):
        format_mapping = {
            "year": "%Y-01-01",  # First day of the year
            "month": "%Y-%m-01",  # First day of the month
            "day": "%Y-%m-%d",  # The exact day
            "week": "%Y-%m-%d",
        }
        return format_mapping.get(interval, "%Y-%m-%d")

    # Replace pattern for months (PostgreSQL to MySQL)
    date_trunc_month_pattern = (
        r"DATE_TRUNC\('month', CURRENT_DATE\) - INTERVAL '(\d+) (months?|days?)'"
    )
    instructions = re.sub(
        date_trunc_month_pattern,
        lambda m: f"DATE_SUB(DATE_FORMAT(CURRENT_DATE, '%Y-%m-01'), INTERVAL {m.group(1)} {m.group(2).upper().rstrip('S')})",
        instructions,
    )

    # Replace pattern for weeks
    date_trunc_week_pattern = (
        r"DATE_TRUNC\('week', CURRENT_DATE\) - INTERVAL '(\d+) (week|weeks)'"
    )
    instructions = re.sub(
        date_trunc_week_pattern,
        lambda m: f"DATE_SUB(DATE_FORMAT(CURRENT_DATE, '%Y-%m-%d'), INTERVAL {int(m.group(1)) * 7} DAY)",
        instructions,
    )

    # Replace pattern for days directly, similar to weeks but simpler
    date_trunc_day_pattern = (
        r"DATE_TRUNC\('day', CURRENT_DATE\) - INTERVAL '(\d+) (day|days)'"
    )
    instructions = re.sub(
        date_trunc_day_pattern,
        lambda m: f"DATE_SUB(CURRENT_DATE, INTERVAL {m.group(1)} DAY)",
        instructions,
    )

    # Replace pattern for DATE_TRUNC without interval
    date_trunc_start_of_week_pattern = r"DATE_TRUNC\('week', CURRENT_DATE\)"
    instructions = re.sub(
        date_trunc_start_of_week_pattern,
        "SUBDATE(CURDATE(), WEEKDAY(CURDATE()))",
        instructions,
    )

    # Replace pattern for DATE_TRUNC using direct equals
    # Example:
    #   PostgreSQL: DATE_TRUNC('month', t1.date) = DATE_TRUNC('month', t2.date)
    #   MySQL: DATE_FORMAT(t1.date, '%Y-%m-01') = DATE_FORMAT(t2.date, '%Y-%m-01')
    date_trunc_date_pattern = (
        r"DATE_TRUNC\('(\w+)', t1.date\)\s*=\s*DATE_TRUNC\('(\w+)', t2.date\)"
    )
    instructions = re.sub(
        date_trunc_date_pattern,
        lambda m: f"DATE_FORMAT(t1.date, '{format_date_for_mysql(m.group(1))}') = DATE_FORMAT(t2.date, '{format_date_for_mysql(m.group(2))}')",
        instructions,
    )

    # Replace pattern for DATE_TRUNC('day', table.datecol)
    date_trunc_pattern = r"DATE_TRUNC\('day', (\w+).(\w+)\)"
    instructions = re.sub(date_trunc_pattern, r"DATE(\1.\2)", instructions)

    # New pattern to handle dynamic DATE_TRUNC in join conditions
    dynamic_date_trunc_join_pattern = r"DATE_TRUNC\('<interval>', (\w+)\.(\w+)\)"
    instructions = re.sub(
        dynamic_date_trunc_join_pattern,
        lambda m: f"DATE_FORMAT({m.group(1)}.{m.group(2)}, '<interval>')",
        instructions,
    )

    # Replace pattern for CURRENT_DATE - INTERVAL '30 days'
    current_date_interval_pattern = r"CURRENT_DATE (-|\+) INTERVAL '(\d+) (day|days)'"
    instructions = re.sub(
        current_date_interval_pattern, r"CURRENT_DATE \1 INTERVAL \2 DAY", instructions
    )

    # Replace pattern for CURRENT_DATE with CURDATE()
    current_date_pattern = r"CURRENT_DATE"
    instructions = re.sub(current_date_pattern, "CURDATE()", instructions)

    # Replace ILIKE for case-insensitive search
    ilike_pattern = r"(\w+)\s+ILIKE\s+'([^']+)'"
    instructions = re.sub(ilike_pattern, r"LOWER(\1) LIKE LOWER('%\2%')", instructions)

    # Generic replacement for ILIKE with LIKE
    instructions = re.sub(r"\b(ILIKE)\b", "LIKE", instructions, flags=re.IGNORECASE)

    # Replace pattern for EPOCH function conversion
    epoch_pattern = r"\bEPOCH\b"
    instructions = re.sub(
        epoch_pattern, "UNIX_TIMESTAMP", instructions, flags=re.IGNORECASE
    )

    return instructions


######## SQLITE FUNCTIONS ########


def sql_to_sqlite(sql, db_type, table_metadata_string):
    """
    Translates sql of db_type to SQLite dialect.
    For SQLite, no extra post-processing is needed. No need to prefix tables with db_name.
    Returns translated sql and translated test sql for testing.
    For SQLite, both the translated sql and test sql are the same.
    """
    translated = sql_to_dialect(sql, db_type, "sqlite")
    translated = sql_remove_schema(translated, table_metadata_string)
    translated_test = translated
    return translated, translated_test


def ddl_to_sqlite(ddl, db_type, db_name, row_idx):
    """
    Translates ddl of db_type to SQLite dialect.
    Returns translated ddl and translated test ddl for testing.
    For SQLite, both the translated ddl and test ddl are the same.
    """
    translated = ddl_to_dialect(ddl, db_type, "sqlite")
    translated = ddl_remove_schema(translated)

    # if any of reserved keywords in the non-comments section of ddl, enclose them with backticks
    reserved_keywords = ["transaction", "order"]
    segments = re.split(r"(/\*.*?\*/)", translated)
    for i, segment in enumerate(segments):
        if not segment.startswith("/*"):
            for keyword in reserved_keywords:
                segment = re.sub(
                    rf'(?<!")\b{keyword}\b(?!")',
                    f"`{keyword}`",
                    segment,
                    flags=re.IGNORECASE,
                )
            segments[i] = segment
    translated = "".join(segments)

    translated = translated.replace(")\nCREATE", ");\nCREATE")
    translated = re.sub(r"SERIAL", "INTEGER PRIMARY KEY", translated)
    translated += ";"
    return translated, translated


def create_sqlite_db(db_name, table_metadata_string_test, row_idx):
    """
    Create a test SQLite database and tables from the table_metadata_string_test.
    """
    test_db_name = f"test{row_idx}_" + db_name
    try:
        conn = sqlite3.connect(f"{test_db_name}.db")
        cursor = conn.cursor()
        for table in table_metadata_string_test.split(");"):
            if table.strip() == "":
                continue
            if not table.endswith(");"):
                table += ");"
            cursor.execute(table)
        # print(f"Tables for `{test_db_name}` created successfully")
    except Exception as err:
        print(f"Error creating database or tables: {err}")
        raise
    finally:
        if "cursor" in locals():
            cursor.close()
        if "conn" in locals():
            conn.close()


def delete_sqlite_db(db_name, row_idx):
    """
    Delete the test SQLite database file.
    """
    test_db_name = f"test{row_idx}_" + db_name + ".db"
    try:
        os.remove(test_db_name)
        # print(f"Database `{test_db_name}` deleted successfully")
    except FileNotFoundError:
        # print(f"Database `{test_db_name}` does not exist")
        pass
    except Exception as err:
        # print(f"Error deleting database `{test_db_name}`: {err}")
        raise


def test_valid_md_sqlite(sql_test_list, db_name, table_metadata_string_test, row_idx):
    """
    Test the validity of the metadata and sql in SQLite.
    This will create a test dataset and tables, run the sql and delete the test dataset.
    """
    validity_tuple_list = []
    test_db = f"test{row_idx}_{db_name}"
    test_db_file = f"{test_db}.db"
    # create a test db
    try:
        create_sqlite_db(db_name, table_metadata_string_test, row_idx)
    except Exception as e:
        if "cursor" in locals():
            cursor.close()
        if "conn" in locals():
            conn.close()
        delete_sqlite_db(db_name, row_idx)
        error_tuple = (False, "Error creating test db: " + str(e))
        validity_tuple_list.extend([error_tuple] * len(sql_test_list))
        return validity_tuple_list

    # run the sql
    for sql_test in sql_test_list:
        tries = 0
        error_msg = ""
        validity_added = False
        while tries < 3 and not validity_added:
            try:
                conn = sqlite3.connect(test_db_file)
                cursor = conn.cursor()

                cursor.execute(sql_test)
                results = cursor.fetchall()
                validity_tuple_list.append((True, ""))
                validity_added = True
            except Exception as e:
                error_msg = str(e)

                if (
                    "no such table" in error_msg
                    or "unable to open database file" in error_msg
                ):
                    print("Error that will lead to retry:", e)
                    print("Retrying...")
                    tries += 1
                    time.sleep(2)
                else:
                    # print("Error running sql:", e)
                    validity_tuple_list.append((False, error_msg))
                    validity_added = True
        if not validity_added:
            validity_tuple_list.append((False, error_msg))

    if "cursor" in locals():
        cursor.close()
    if "conn" in locals():
        conn.close()
    delete_sqlite_db(db_name, row_idx)
    return validity_tuple_list


def test_valid_md_sqlite_concurr(df, sql_list_col, table_metadata_col):
    """
    Run test_valid_md_sqlite concurrently on a DataFrame
    """

    futures_to_index = {}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Prepare futures for each row in the DataFrame
        futures = [
            executor.submit(
                test_valid_md_sqlite,
                row[sql_list_col],
                row["db_name"],
                row[table_metadata_col],
                row.get("index", str(index)),
            )
            for index, row in df.iterrows()
        ]

        # Map each future to its index
        for index, future in enumerate(futures):
            futures_to_index[future] = index

        # Collect results as they complete and map them back to their indices
        results = [None] * len(futures)
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            index = futures_to_index[future]
            results[index] = future.result()

    return results


def test_valid_sqlite(creds, sql_test_list, db_name):
    """
    Test the validity of the sql on an existing DB in SQLite.
    If the sql is valid and returns non-empty results, return True.
    """
    validity_tuple_list = []
    db_file = creds["path_to_folder"] + f"{db_name}.db"

    # run the sql
    for sql_test in sql_test_list:
        error_msg = ""
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            cursor.execute(sql_test)
            results = cursor.fetchall()
            # check if results empty
            if results:
                validity_tuple_list.append((True, ""))
            else:
                validity_tuple_list.append(
                    (
                        False,
                        "Results are empty, which means SQL is wrong and does not answer the question. Or data values in defog-data databases could be outdated.",
                    )
                )
        except Exception as e:
            error_msg = str(e)
            validity_tuple_list.append((False, error_msg))

    if "cursor" in locals():
        cursor.close()
    if "conn" in locals():
        conn.close()
    return validity_tuple_list


def instructions_to_sqlite(instructions):
    """
    Convert the db-specific instructions to SQLite dialect.
    """
    # Replace pattern for months (example included in the original pattern)
    date_trunc_month_pattern = (
        r"DATE_TRUNC\('month', CURRENT_DATE\) - INTERVAL '(\d+) (months?|days?)'"
    )
    instructions = re.sub(
        date_trunc_month_pattern,
        r"DATE('now', 'start of month', '-\1 \2')",
        instructions,
    )

    # Replace pattern for weeks
    date_trunc_week_pattern = (
        r"DATE_TRUNC\('week', CURRENT_DATE\) - INTERVAL '(\d+) (week|weeks)'"
    )
    instructions = re.sub(
        date_trunc_week_pattern,
        lambda m: f"DATE('now',  '-' || ((strftime('%w', 'now') + 6) % 7) || ' days', '-{int(m.group(1)) * 7} days')",
        instructions,
    )

    # Replace pattern for days
    date_trunc_day_pattern = (
        r"DATE_TRUNC\('week', CURRENT_DATE\) - INTERVAL '(\d+) (day|days)'"
    )
    instructions = re.sub(
        date_trunc_day_pattern,
        lambda m: f"DATE('now',  '-' || ((strftime('%w', 'now') + 6) % 7) || ' days', '-{m.group(1)} {m.group(2)}')",
        instructions,
    )

    # Replace pattern for DATE_TRUNC without interval
    date_trunc_nointerval_pattern = r"DATE_TRUNC\('week', CURRENT_DATE\)"
    instructions = re.sub(
        date_trunc_nointerval_pattern,
        lambda m: f"DATE('now',  '-' || ((strftime('%w', 'now') + 6) % 7) || ' days')",
        instructions,
    )

    # Replace pattern for DATE_TRUNC('<interval>', t1.date)=DATE_TRUNC('<interval>', t2.date)
    date_trunc_date_pattern = (
        r"DATE_TRUNC\('(\<\w+\>)', t1.date\)\s*=\s*DATE_TRUNC\('(\<\w+\>)', t2.date\)"
    )
    instructions = re.sub(
        date_trunc_date_pattern,
        r"DATE(t1.date, '\1') = DATE(t2.date, '\2')",
        instructions,
    )

    # Replace pattern for DATE_TRUNC('day', table.datecol)
    date_trunc_pattern = r"DATE_TRUNC\('day', (\w+).(\w+)\)"
    instructions = re.sub(date_trunc_pattern, r"DATETIME(DATE(\1.\2))", instructions)

    # Replace pattern for CURRENT_DATE - INTERVAL '30 days'
    current_date_interval_pattern = r"CURRENT_DATE (-|\+) INTERVAL '(\d+) (day|days)'"
    instructions = re.sub(
        current_date_interval_pattern, r"DATE('now', \1'\2 \3')", instructions
    )

    # Replace pattern for CURRENT_DATE with DATE('now')
    current_date_pattern = r"CURRENT_DATE"
    instructions = re.sub(current_date_pattern, r"DATE('now')", instructions)

    return instructions


######## T-SQL FUNCTIONS ########


def sql_to_tsql(sql, db_type):
    """
    Translates sql of db_type to T-SQL dialect.
    For T-SQL, no extra post-processing is needed. No need to prefix tables with db_name.
    Returns translated sql and translated test sql for testing.
    For T-SQL, both the translated sql and test sql are the same.
    """
    translated = sql_to_dialect(sql, db_type, "tsql")
    translated_test = translated
    return translated, translated_test


def ddl_to_tsql(ddl, db_type, db_name, row_idx):
    """
    Translates ddl of db_type to T-SQL dialect.
    Returns translated ddl and translated test ddl for testing.
    For T-SQL, both the translated ddl and test ddl are the same.
    """
    translated = ddl_to_dialect(ddl, db_type, "tsql")
    translated = translated.replace(")\nCREATE", ");\nCREATE")
    translated = re.sub(r"SERIAL(PRIMARY KEY)?", "INT IDENTITY(1,1)", translated)
    translated += ";"
    return translated, translated


def create_tsql_db(creds, db_name, table_metadata_string_test, row_idx):
    """
    Create a test T-SQL database and tables from the table_metadata_string
    """
    import pyodbc

    test_db_name = f"test{row_idx}_" + db_name
    try:
        with pyodbc.connect(
            f"DRIVER={creds['driver']};SERVER={creds['server']};UID={creds['user']};PWD={creds['password']}"
        ) as conn:
            conn.autocommit = True
            with conn.cursor() as cursor:
                cursor.execute(f"CREATE DATABASE {test_db_name};")
                # print(f"Database `{test_db_name}` created successfully")
    except pyodbc.Error as err:
        if err.args[0] == "28000":
            print("Something is wrong with your user name or password")
        elif err.args[0] == "42000":
            print(f"Database `{test_db_name}` already exists")
            pass
        elif "Communication link failure" in str(err):
            time.sleep(2)
            print(f"Lost connection for `{test_db_name}`. Retrying...")
            create_tsql_db(creds, db_name, table_metadata_string_test, row_idx)
        else:
            print(err)
        raise

    try:
        with pyodbc.connect(
            f"DRIVER={creds['driver']};SERVER={creds['server']};DATABASE={test_db_name};UID={creds['user']};PWD={creds['password']}"
        ) as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"USE {test_db_name};")
                cursor.execute(table_metadata_string_test)
                # print(f"Tables for `{test_db_name}` created successfully")
    except Exception as e:
        print(f"Error creating tables for `{test_db_name}`: {e}")
        raise
    finally:
        if "cursor" in locals():
            cursor.close()
        if "conn" in locals():
            conn.close()


def delete_tsql_db(creds, db_name, row_idx):
    """
    Delete a test T-SQL database
    """
    import pyodbc

    test_db_name = f"test{row_idx}_{db_name}"
    try:
        with pyodbc.connect(
            f"DRIVER={creds['driver']};SERVER={creds['server']};DATABASE=master;UID={creds['user']};PWD={creds['password']}"
        ) as conn:
            conn.autocommit = True
            with conn.cursor() as cursor:
                cursor.execute(f"DROP DATABASE {test_db_name};")
                # print(f"Database {test_db_name} deleted successfully")
    except pyodbc.Error as err:
        if err.args[0] == "28000":
            print("Something is wrong with your user name or password")
        elif err.args[0] == "42S02":
            # print(f"Database `{test_db_name}` does not exist")
            pass
        else:
            print(f"Unexpected error deleting `{test_db_name}`: {err}")


def test_valid_md_tsql(
    creds, sql_test_list, db_name, table_metadata_string_test, row_idx
):
    """
    Test the validity of the metadata and sql in T-SQL.
    This will create a test dataset and tables, run the sql and delete the test dataset.
    """
    import pyodbc

    validity_tuple_list = []
    test_db = f"test{row_idx}_{db_name}"
    # create a test db
    try:
        create_tsql_db(creds, db_name, table_metadata_string_test, row_idx)
    except Exception as e:
        if "cursor" in locals():
            cursor.close()
        if "conn" in locals():
            conn.close()
        time.sleep(2)
        delete_tsql_db(creds, db_name, row_idx)
        error_tuple = (False, "Error creating test db: " + str(e))
        validity_tuple_list.extend([error_tuple] * len(sql_test_list))
        return validity_tuple_list

    # run the sql
    for sql_test in sql_test_list:
        tries = 0
        error_msg = ""
        validity_added = False
        while tries < 3 and not validity_added:
            try:
                conn = pyodbc.connect(
                    f"DRIVER={creds['driver']};SERVER={creds['server']};DATABASE={test_db};UID={creds['user']};PWD={creds['password']}"
                )
                cursor = conn.cursor()

                use_db = f"USE {test_db};"
                cursor.execute(use_db)
                cursor.execute(sql_test)
                results = cursor.fetchall()
                time.sleep(2)
                validity_tuple_list.append((True, ""))
                validity_added = True
            except Exception as e:
                error_msg = str(e)

                time.sleep(2)
                if "Invalid table" in error_msg or "Invalid column" in error_msg:
                    tries += 1
                else:
                    # print("Error running sql:", e)
                    validity_tuple_list.append((False, error_msg))
                    validity_added = True
        if not validity_added:
            validity_tuple_list.append((False, error_msg))

    if "cursor" in locals():
        cursor.close()
    if "conn" in locals():
        conn.close()
    time.sleep(2)
    delete_tsql_db(creds, db_name, row_idx)
    return validity_tuple_list


def test_valid_md_tsql_concurr(df, db_creds_all, sql_list_col, table_metadata_col):
    """
    Run test_valid_md_tsql concurrently on a DataFrame
    """

    futures_to_index = {}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Prepare futures for each row in the DataFrame
        futures = [
            executor.submit(
                test_valid_md_tsql,
                db_creds_all["tsql"],
                row[sql_list_col],
                row["db_name"],
                row[table_metadata_col],
                row.get("index", str(index)),
            )
            for index, row in df.iterrows()
        ]

        # Map each future to its index
        for index, future in enumerate(futures):
            futures_to_index[future] = index

        # Collect results as they complete and map them back to their indices
        results = [None] * len(futures)
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            index = futures_to_index[future]
            results[index] = future.result()

    return results


def test_valid_tsql(creds, sql_test_list, db_name):
    """
    Test the validity of the sql on an existing DB in T-SQL.
    If the sql is valid and returns non-empty results, return True.
    """
    import pyodbc

    validity_tuple_list = []
    # run the sql
    for sql_test in sql_test_list:
        error_msg = ""
        try:
            conn = pyodbc.connect(
                f"DRIVER={creds['driver']};SERVER={creds['server']};DATABASE={db_name};UID={creds['user']};PWD={creds['password']}"
            )
            cursor = conn.cursor()
            use_db = f"USE {db_name};"
            cursor.execute(use_db)
            cursor.execute(sql_test)
            results = cursor.fetchall()
            # check if results empty
            if results:
                validity_tuple_list.append((True, ""))
            else:
                validity_tuple_list.append(
                    (
                        False,
                        "Results are empty, which means SQL is wrong and does not answer the question. Or data values in defog-data databases could be outdated.",
                    )
                )
        except Exception as e:
            error_msg = str(e)
            validity_tuple_list.append((False, error_msg))

    if "cursor" in locals():
        cursor.close()
    if "conn" in locals():
        conn.close()
    return validity_tuple_list


def instructions_to_tsql(instructions):
    """
    Convert the db-specific instructions to T-SQL dialect.
    """

    def remove_last_char_if_s(s):
        if s.endswith("S"):
            return s[:-1]
        return s

    # Replace pattern for months (example included in the original pattern)
    date_trunc_month_pattern = (
        r"DATE_TRUNC\('month', CURRENT_DATE\) - INTERVAL '(\d+) (months?|days?)'"
    )
    instructions = re.sub(
        date_trunc_month_pattern,
        lambda m: f"DATEADD({remove_last_char_if_s(m.group(2).upper())}, -{m.group(1)}, DATEADD(MONTH, DATEDIFF(MONTH, 0, GETDATE()), 0))",
        instructions,
    )

    # Replace pattern for weeks
    date_trunc_week_pattern = (
        r"DATE_TRUNC\('week', CURRENT_DATE\) - INTERVAL '(\d+) (week|weeks)'"
    )
    instructions = re.sub(
        date_trunc_week_pattern,
        lambda m: f"DATEADD(WEEK, -{m.group(1)}, DATEADD(WEEK, DATEDIFF(WEEK, 0, GETDATE()), 0))",
        instructions,
    )

    # Replace pattern for days
    date_trunc_day_pattern = (
        r"DATE_TRUNC\('week', CURRENT_DATE\) - INTERVAL '(\d+) (day|days)'"
    )
    instructions = re.sub(
        date_trunc_day_pattern,
        lambda m: f"DATEADD(DAY, -{m.group(1)}, DATEADD(WEEK, DATEDIFF(WEEK, 0, GETDATE()), 0))",
        instructions,
    )

    # Replace pattern for DATE_TRUNC without interval
    date_trunc_nointerval_pattern = r"DATE_TRUNC\('week', CURRENT_DATE\)"
    instructions = re.sub(
        date_trunc_nointerval_pattern,
        lambda m: f"DATEADD(WEEK, DATEDIFF (WEEK, 0, GETDATE()), 0)",
        instructions,
    )

    # Replace pattern for DATE_TRUNC('<interval>', t1.date)=DATE_TRUNC('<interval>', t2.date)
    date_trunc_date_pattern = (
        r"DATE_TRUNC\('(\<\w+\>)', t1.date\)\s*=\s*DATE_TRUNC\('(\<\w+\>)', t2.date\)"
    )
    instructions = re.sub(
        date_trunc_date_pattern,
        r"DATEADD('\1', DATEDIFF('\1', 0, t1.date), 0) = DATEADD('\2', DATEDIFF('\2', 0, t2.date), 0)",
        instructions,
    )

    # Replace pattern for DATE_TRUNC('day', table.datecol)
    date_trunc_pattern = r"DATE_TRUNC\('day', (\w+).(\w+)\)"
    instructions = re.sub(
        date_trunc_pattern, r"CONVERT(DATETIME, CONVERT(DATE, \1.\2))", instructions
    )

    # Replace pattern for CURRENT_DATE - INTERVAL '30 days'
    current_date_interval_pattern = r"CURRENT_DATE (-|\+) INTERVAL '(\d+) (day|days)'"
    instructions = re.sub(
        current_date_interval_pattern,
        r"DATEADD(DAY, \1\2, CAST(GETDATE() AS DATE))",
        instructions,
    )

    # Replace pattern for CURRENT_DATE
    current_date_pattern = r"CURRENT_DATE"
    instructions = re.sub(
        current_date_pattern, r"CAST(GETDATE() AS DATE)", instructions
    )

    return instructions


### General conversion function
def convert_postgres_ddl_to_dialect(postgres_ddl: str, to_dialect: str, db_name: str):
    """
    This function converts a ddl from postgres to another dialect.
    We have a separate function for this since the default defog_data DDLS
    are for Postgres, and using this means less code when converting.
    """
    if to_dialect == "postgres":
        return postgres_ddl
    elif to_dialect == "bigquery":
        new_ddl, _ = ddl_to_bigquery(postgres_ddl, "postgres", db_name, 42)
    elif to_dialect == "mysql":
        new_ddl, _ = ddl_to_mysql(postgres_ddl, "postgres", db_name, 42)
    elif to_dialect == "sqlite":
        new_ddl, _ = ddl_to_sqlite(postgres_ddl, "postgres", db_name, 42)
    elif to_dialect == "tsql":
        new_ddl, _ = ddl_to_tsql(postgres_ddl, "postgres", db_name, 42)
    else:
        raise ValueError(f"Unsupported dialect {to_dialect}")
    return new_ddl
