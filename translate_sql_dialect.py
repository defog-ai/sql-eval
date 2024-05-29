import pandas as pd
import asyncio
import json
from utils.dialects import (
    sql_to_bigquery,
    ddl_to_bigquery,
    test_valid_md_bq_concurr,
    amend_invalid_sql_concurr,
    sql_to_mysql,
    ddl_to_mysql,
    test_valid_md_mysql_concurr,
    sql_to_sqlite,
    ddl_to_sqlite,
    test_valid_md_sqlite_concurr,
    sql_to_tsql,
    ddl_to_tsql,
    test_valid_md_tsql_concurr,
)
from utils.gen_prompt import to_prompt_schema
from tqdm import tqdm
import os

tqdm.pandas()

dataset_file = "data/instruct_basic_postgres.csv" # Postgres dataset file to translate
dialect = "bigquery"  # Supported dialects: "bigquery", "mysql", "sqlite", "tsql"
bigquery_proj = os.getenv(
    "BIGQUERY_PROJ"
)  # Set this to your BigQuery project ID, leave empty if dialect is not BigQuery

model = "gpt-4o"  # Model to use for translation of invalid SQL
max_concurrent = 5  # Maximum number of concurrent coroutines when querying openai
output_file = dataset_file.replace("postgres", dialect)

df = pd.read_csv(dataset_file)

# set validity and error_msg to empty strings
df["valid"] = ""
df["err_msg"] = ""

# create db_type col where if "Snowflake" in instructions of row, db_type = "Snowflake" else "postgres"
if "snowflake" in dataset_file:
    df["db_type"] = "snowflake"
else:
    df["db_type"] = "postgres"

# if db_name is empty, use "dbname"
df["db_name"] = df.apply(
    lambda x: "dbname"
    if (pd.isna(x.get("db_name")) and x.get("db_name") != "")
    else (x["db_name"]),
    axis=1,
)


# get full table_metadata_string for all rows
def get_md_string(db_name):
    """
    Get the table metadata string from the metadata dictionary.
    """
    from defog_data.metadata import dbs

    md = dbs[db_name]["table_metadata"]
    table_metadata_string = to_prompt_schema(md)
    return table_metadata_string


df["table_metadata_string"] = df.progress_apply(
    lambda x: get_md_string(x["db_name"]), axis=1
)

############################ Translation ############################

# translate query col to dialect with sqlglot
print(f"Translating all SQL to {dialect} with sqlglot...")
if dialect == "bigquery":
    df[f"sql_tuple"] = df.progress_apply(
        lambda x: sql_to_bigquery(
            x["query"],
            x["db_type"],
            x["table_metadata_string"],
            x["db_name"],
            str(x.name),
        ),
        axis=1,
    )
elif dialect == "mysql":
    df[f"sql_tuple"] = df.progress_apply(
        lambda x: sql_to_mysql(
            x["query"],
            x["db_type"],
            x["table_metadata_string"],
        ),
        axis=1,
    )
elif dialect == "sqlite":
    df[f"sql_tuple"] = df.progress_apply(
        lambda x: sql_to_sqlite(
            x["query"],
            x["db_type"],
            x["table_metadata_string"],
        ),
        axis=1,
    )
elif dialect == "tsql":
    df[f"sql_tuple"] = df.progress_apply(
        lambda x: sql_to_tsql(
            x["query"],
            x["db_type"],
        ),
        axis=1,
    )
df[f"sql_{dialect}"], df[f"sql_{dialect}_test"] = zip(*df["sql_tuple"])
# df = df[df[f"sql_{dialect}"].notnull()]  # drop rows where sql_dialect is None
df.drop(columns=["sql_tuple"], inplace=True)

# translate ddl col to dialect
print(f"Translating all DDL to {dialect}...")
if dialect == "bigquery":
    df[f"table_metadata_string_tuple"] = df.progress_apply(
        lambda x: ddl_to_bigquery(
            x["table_metadata_string"],
            x["db_type"],
            x["db_name"],
            str(x.name),
        ),
        axis=1,
    )
elif dialect == "mysql":
    df[f"table_metadata_string_tuple"] = df.progress_apply(
        lambda x: ddl_to_mysql(
            x["table_metadata_string"],
            x["db_type"],
            x["db_name"],
            str(x.name),
        ),
        axis=1,
    )
elif dialect == "sqlite":
    df[f"table_metadata_string_tuple"] = df.progress_apply(
        lambda x: ddl_to_sqlite(
            x["table_metadata_string"],
            x["db_type"],
            x["db_name"],
            str(x.name),
        ),
        axis=1,
    )
elif dialect == "tsql":
    df[f"table_metadata_string_tuple"] = df.progress_apply(
        lambda x: ddl_to_tsql(
            x["table_metadata_string"],
            x["db_type"],
            x["db_name"],
            str(x.name),
        ),
        axis=1,
    )
df[f"table_metadata_{dialect}"], df[f"table_metadata_{dialect}_test"] = zip(
    *df["table_metadata_string_tuple"]
)
df.drop(columns=["table_metadata_string_tuple"], inplace=True)

############################ Validity Check ############################

# run test_valid_md_bq_concurr concurrently on the DataFrame
print(f"Checking validity of all translated SQL and DDL in {dialect}...")
sql_col = f"sql_{dialect}_test"
table_metadata_col = f"table_metadata_{dialect}_test"
if dialect == "bigquery":
    df["result_tuple"] = test_valid_md_bq_concurr(
        df, bigquery_proj, sql_col, table_metadata_col
    )
elif dialect == "mysql":
    df["result_tuple"] = test_valid_md_mysql_concurr(df, sql_col, table_metadata_col)
elif dialect == "sqlite":
    df["result_tuple"] = test_valid_md_sqlite_concurr(df, sql_col, table_metadata_col)
elif dialect == "tsql":
    df["result_tuple"] = test_valid_md_tsql_concurr(df, sql_col, table_metadata_col)
df["valid"], df["err_msg"] = zip(*df["result_tuple"])
df.drop(columns=["result_tuple"], inplace=True)
df.reset_index(inplace=True)
df_invalid = df[df["valid"] == False].copy()
print("No. of invalid rows: ", len(df_invalid))

############################ Correction ############################

# use llm to correct invalid SQL if any
if df_invalid.shape[0] > 0:
    print(f"Correcting invalid SQL using {model}...")

    async def main():
        results = await amend_invalid_sql_concurr(
            df_invalid, model, max_concurrent, dialect
        )
        df_invalid["corrected_sql"] = results

    asyncio.run(main())

    # extract corrected SQL and add to DataFrame
    df_invalid[f"sql_{dialect}_test_corrected"] = df_invalid["corrected_sql"].apply(
        lambda x: x.get("sql")
    )
    df_invalid[f"sql_{dialect}_corrected"] = df_invalid.apply(
        lambda row: row[f"sql_{dialect}_test_corrected"].replace(
            f"test{row.get('index', row.name)}_", ""
        ),
        axis=1,
    )
    df_invalid.drop(columns=["corrected_sql"], inplace=True)

    # check validity of corrected SQL
    print(f"Checking validity of corrected SQL in {dialect}...")
    sql_col = f"sql_{dialect}_test_corrected"
    table_metadata_col = f"table_metadata_{dialect}_test"
    if dialect == "bigquery":
        df_invalid["result_tuple"] = test_valid_md_bq_concurr(
            df_invalid, bigquery_proj, sql_col, table_metadata_col
        )
    elif dialect == "mysql":
        df_invalid["result_tuple"] = test_valid_md_mysql_concurr(
            df_invalid, sql_col, table_metadata_col
        )
    elif dialect == "sqlite":
        df_invalid["result_tuple"] = test_valid_md_sqlite_concurr(
            df_invalid, sql_col, table_metadata_col
        )
    elif dialect == "tsql":
        df_invalid["result_tuple"] = test_valid_md_tsql_concurr(
            df_invalid, sql_col, table_metadata_col
        )
    df_invalid["valid"], df_invalid["err_msg"] = zip(*df_invalid["result_tuple"])
    df_invalid.drop(columns=["result_tuple"], inplace=True)

    # get valid from df_invalid
    df_corrected_valid = df_invalid[df_invalid["valid"] == True].copy()
    print("No. of corrected valid rows: ", len(df_corrected_valid))

    # replace sqlglot translated columns with LLM corrected columns
    df_corrected_valid.drop(
        columns=[f"sql_{dialect}", f"sql_{dialect}_test"], inplace=True
    )
    df_corrected_valid.rename(
        columns={
            f"sql_{dialect}_corrected": f"sql_{dialect}",
            f"sql_{dialect}_test_corrected": f"sql_{dialect}_test",
        },
        inplace=True,
    )

    # merge corrected valid rows with original DataFrame
    merged_df = pd.concat([df, df_corrected_valid], ignore_index=False, axis=0)
    # deduplicate indices in merged_df and keep only corrected rows
    merged_df = merged_df.loc[~merged_df.index.duplicated(keep="last")]
    merged_df = merged_df.copy()
    merged_df.sort_index(inplace=True)
else:
    merged_df = df.copy()

############################ Post-Processing ############################

# count no. of invalid rows
n_invalid = len(merged_df[merged_df["valid"] == False])
print("No. of invalid rows remaining: ", n_invalid)
if n_invalid > 0:
    print("Please manually correct the invalid SQL(s) in the output file.")

# prefix all invalid sql with "INVALID: err_msg"
merged_df[f"sql_{dialect}"] = merged_df.apply(
    lambda x: f"INVALID: {x['err_msg']}-----------------{x[f'sql_{dialect}']}-----------------Original postgres: {x['query']}"
    if x["valid"] == False
    else x[f"sql_{dialect}"],
    axis=1,
)

merged_df.fillna("", inplace=True)

# change all db_type to dialect
merged_df["db_type"] = dialect

# drop original query col and table_metadata_string col
merged_df.drop(columns=["query", "table_metadata_string"], inplace=True)

# rename sql_{dialect} to sql, table_metadata_{dialect} to table_metadata_string
merged_df.rename(
    columns={
        f"sql_{dialect}": "query",
    },
    inplace=True,
)

# drop cols
drop_columns = [
    "valid",
    "err_msg",
    f"sql_{dialect}_test",
    f"table_metadata_{dialect}_test",
    f"table_metadata_{dialect}",
    "index",
]
merged_df.drop(columns=drop_columns, inplace=True)


# reorder cols
first_cols = [
    "db_name",
    "db_type",
    "query_category",
    "query",
    "question",
]
cols = list(merged_df.columns)
cols = first_cols + [col for col in cols if col not in first_cols]
merged_df = merged_df[cols]

# save to json
merged_df.to_csv(output_file, index=False)
print(f"Saved to {output_file}")
