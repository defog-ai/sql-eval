import pandas as pd
import asyncio
import json
from utils.dialects import (
    sql_to_bigquery,
    ddl_to_bigquery,
    test_valid_bq,
    amend_invalid_sql_concurr,
    sql_to_mysql,
    ddl_to_mysql,
    test_valid_mysql,
    sql_to_sqlite,
    ddl_to_sqlite,
    instructions_to_sqlite,
    instructions_to_mysql,
    instructions_to_tsql,
    test_valid_sqlite,
    sql_to_tsql,
    ddl_to_tsql,
    test_valid_tsql,
    get_schema_names,
)
from utils.gen_prompt import to_prompt_schema
from utils.creds import db_creds_all
from tqdm import tqdm
from eval.eval import get_all_minimal_queries
import os

tqdm.pandas()

dataset_file = (
    "data/instruct_advanced_postgres.csv"  # Postgres dataset file to translate
)
dialect = "mysql"  # Supported dialects: "bigquery", "mysql", "sqlite", "tsql"
model = "gpt-4-turbo"  # Model to use for translation of invalid SQL
max_concurrent = 5  # Maximum number of concurrent coroutines when querying openai
if "postgres" in dataset_file:
    output_file = dataset_file.replace("postgres", dialect)
else:
    output_file = dataset_file.replace(".csv", f"_{dialect}.csv")

df = pd.read_csv(dataset_file)

# set validity and error_msg to empty strings
df["valid_list"] = ""
df["err_msg_list"] = ""

# fill na with empty string
df.fillna("", inplace=True)

# create db_type col where if "Snowflake" in file name, db_type = "snowflake", else db_type = "postgres"
if "snowflake" in dataset_file:
    df["db_type"] = "snowflake"
else:
    df["db_type"] = "postgres"

# if ILIKE in instructions col, and db_type is in ["sqlite", "bigquery", "tsql"], replace ILIKE with LIKE
if "instructions" in df.columns:
    df["instructions"] = df["instructions"].apply(
        lambda x: (
            x.replace("ILIKE", "LIKE")
            if "ILIKE" in x and dialect in ["sqlite", "bigquery", "tsql"]
            else x
        )
    )
# translate instructions and full_instructions columns to dialect
if "instructions" in df.columns:
    if dialect == "sqlite":
        df["instructions"] = df.progress_apply(
            lambda x: instructions_to_sqlite(x["instructions"]), axis=1
        )
    elif dialect == "tsql":
        df["instructions"] = df.progress_apply(
            lambda x: instructions_to_tsql(x["instructions"]), axis=1
        )
    elif dialect == "mysql":
        df["instructions"] = df.progress_apply(
            lambda x: instructions_to_mysql(x["instructions"]), axis=1
        )
    else:
        print(
            f"Instructions translation not yet supported for {dialect}. Please add an instructions_to_{dialect} function in utils/dialects.py"
        )

if "full_instructions" in df.columns:
    if dialect == "sqlite":
        df["full_instructions"] = df.progress_apply(
            lambda x: instructions_to_sqlite(x["full_instructions"]), axis=1
        )
    elif dialect == "tsql":
        df["full_instructions"] = df.progress_apply(
            lambda x: instructions_to_tsql(x["full_instructions"]), axis=1
        )
    else:
        print(
            f"Instructions translation not yet supported for {dialect}. Please add an instructions_to_{dialect} function in utils/dialects.py"
        )

# if db_name is empty, use "dbname"
df["db_name"] = df.apply(
    lambda x: (
        "dbname"
        if (pd.isna(x.get("db_name")) and x.get("db_name") != "")
        else (x["db_name"])
    ),
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
    # add CREATE SCHEMA statements if schema names are present
    schema_names = get_schema_names(table_metadata_string)
    if schema_names:
        for schema_name in schema_names:
            table_metadata_string = (
                f"CREATE SCHEMA IF NOT EXISTS {schema_name};\n" + table_metadata_string
            )
    return table_metadata_string


df["table_metadata_string"] = df.progress_apply(
    lambda x: get_md_string(x["db_name"]), axis=1
)


# remove `schema_name.` from instructions from all rows if dialect is in ["sqlite", "bigquery", "mysql"]
def remove_schema_instructions(table_metadata_string, instructions):
    schema_names = get_schema_names(table_metadata_string)
    for schema_name in schema_names:
        instructions = instructions.replace(f"{schema_name}.", "")
    return instructions


if "instructions" in df.columns:
    df["instructions"] = df.progress_apply(
        lambda x: (
            remove_schema_instructions(x["table_metadata_string"], x["instructions"])
            if dialect in ["sqlite", "bigquery", "mysql"]
            else x["instructions"]
        ),
        axis=1,
    )

# get all minimal queries for all rows
df["query_list"] = df.progress_apply(
    lambda x: get_all_minimal_queries(x["query"]), axis=1
)

############################ Translation ############################

# translate query col to dialect with sqlglot
print(f"Translating all SQL to {dialect} with sqlglot...")
if dialect == "bigquery":
    df["sql_tuple_list"] = df.progress_apply(
        lambda x: [
            sql_to_bigquery(
                query,
                x["db_type"],
                x["table_metadata_string"],
                x["db_name"],
                str(x.name),
            )
            for query in x["query_list"]
        ],
        axis=1,
    )
elif dialect == "mysql":
    df["sql_tuple_list"] = df.progress_apply(
        lambda x: [
            sql_to_mysql(
                query,
                x["db_type"],
                x["table_metadata_string"],
            )
            for query in x["query_list"]
        ],
        axis=1,
    )

elif dialect == "sqlite":
    df["sql_tuple_list"] = df.progress_apply(
        lambda x: [
            sql_to_sqlite(
                query,
                x["db_type"],
                x["table_metadata_string"],
            )
            for query in x["query_list"]
        ],
        axis=1,
    )
elif dialect == "tsql":
    df["sql_tuple_list"] = df.progress_apply(
        lambda x: [
            sql_to_tsql(
                query,
                x["db_type"],
            )
            for query in x["query_list"]
        ],
        axis=1,
    )
# create sql_dialect_list (list of first items in tuple)
df[f"sql_{dialect}_list"] = df["sql_tuple_list"].apply(
    lambda x: [item[0] for item in x]
)
df.drop(columns=["sql_tuple_list"], inplace=True)

# translate ddl col to dialect (only for use in amending invalid SQL)
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
df[f"table_metadata_{dialect}"], _ = zip(*df["table_metadata_string_tuple"])
df.drop(columns=["table_metadata_string_tuple"], inplace=True)

###################### Validity Check on Test DBs ##########################

# test the validity of the queries on the defog-data DBs sequentially
print(f"Checking validity of all translated SQL on existing DBs in {dialect}...")

df["result_tuple_list"] = ""
sql_col = f"sql_{dialect}_list"
for i, row in tqdm(df.iterrows(), total=len(df)):
    sql_list = row[sql_col]
    if dialect == "bigquery":
        result_tuple_list = test_valid_bq(
            db_creds_all["bigquery"], sql_list, row.db_name
        )
    elif dialect == "mysql":
        result_tuple_list = test_valid_mysql(
            db_creds_all["mysql"], sql_list, row.db_name
        )
    elif dialect == "sqlite":
        result_tuple_list = test_valid_sqlite(
            db_creds_all["sqlite"], sql_list, row.db_name
        )
    elif dialect == "tsql":
        result_tuple_list = test_valid_tsql(db_creds_all["tsql"], sql_list, row.db_name)
    else:
        raise ValueError("Dialect not supported")
    df.at[i, "result_tuple_list"] = result_tuple_list

df[f"valid_list"] = df["result_tuple_list"].apply(lambda x: [item[0] for item in x])
df[f"err_msg_list"] = df["result_tuple_list"].apply(lambda x: [item[1] for item in x])

df.drop(columns=["result_tuple_list"], inplace=True)
df.reset_index(inplace=True)

# get rows with at least one invalid SQL
df_invalid = df[df["valid_list"].apply(lambda x: False in x)].copy()
print("No. of invalid rows: ", len(df_invalid))

############################ Correction ############################

# use llm to correct invalid SQL if any
if df_invalid.shape[0] > 0:
    print(f"Correcting invalid SQL using {model}...")

    async def main():
        results = await amend_invalid_sql_concurr(
            df_invalid, model, max_concurrent, dialect
        )
        df_invalid["corrected_sql_list"] = results

    asyncio.run(main())

    # extract corrected SQL and add to DataFrame
    df_invalid[f"sql_{dialect}_corrected_list"] = df_invalid[
        "corrected_sql_list"
    ].apply(lambda x: [item.get("sql") for item in x])

    df_invalid.drop(columns=["corrected_sql_list"], inplace=True)

    # check validity of corrected SQL
    print(f"Checking validity of corrected SQL in {dialect}...")
    sql_col = f"sql_{dialect}_corrected_list"
    df_invalid["result_tuple_list"] = ""

    for i, row in tqdm(df_invalid.iterrows(), total=len(df_invalid)):
        sql_list = row[sql_col]
        if dialect == "bigquery":
            result_tuple_list = test_valid_bq(
                db_creds_all["bigquery"], sql_list, row.db_name
            )
        elif dialect == "mysql":
            result_tuple_list = test_valid_mysql(
                db_creds_all["mysql"], sql_list, row.db_name
            )
        elif dialect == "sqlite":
            result_tuple_list = test_valid_sqlite(
                db_creds_all["sqlite"], sql_list, row.db_name
            )
        elif dialect == "tsql":
            result_tuple_list = test_valid_tsql(
                db_creds_all["tsql"], sql_list, row.db_name
            )
        else:
            raise ValueError("Dialect not supported")
        df_invalid.at[i, "result_tuple_list"] = result_tuple_list

    df_invalid[f"valid_list"] = df_invalid["result_tuple_list"].apply(
        lambda x: [item[0] for item in x]
    )
    df_invalid[f"err_msg_list"] = df_invalid["result_tuple_list"].apply(
        lambda x: [item[1] for item in x]
    )
    df_invalid.drop(columns=["result_tuple_list"], inplace=True)

    # get corrected valid rows where all SQLs are valid
    df_corrected_valid = df_invalid[
        df_invalid["valid_list"].apply(lambda x: False not in x)
    ].copy()
    print("No. of corrected valid rows: ", len(df_corrected_valid))

    # replace sqlglot translated columns with LLM corrected columns
    df_corrected_valid.drop(columns=[f"sql_{dialect}_list"], inplace=True)
    df_corrected_valid.rename(
        columns={
            f"sql_{dialect}_corrected_list": f"sql_{dialect}_list",
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

# count no. of invalid rows where there is at least one invalid SQL
n_invalid = len(merged_df[merged_df["valid_list"].apply(lambda x: False in x)])
print("No. of invalid rows remaining: ", n_invalid)
if n_invalid > 0:
    print("Please manually correct the invalid SQL(s) in the output file.")

# prefix all invalid sql with "INVALID: err_msg"
merged_df[f"sql_{dialect}_list"] = merged_df.apply(
    lambda row: [
        (
            f"<INVALID ERR MSG>: {row['err_msg_list'][index]}-----------------<INVALID TRANSLATION>: {item}-----------------<ORIG POSTGRES>: {row['query_list'][index]}-----------------"
            if row["valid_list"][index] == False
            else item
        )
        for index, item in enumerate(row[f"sql_{dialect}_list"])
    ],
    axis=1,
)
# join all SQLs in the list to a single string and add "; to the last SQL
merged_df[f"sql_{dialect}_list"] = merged_df[f"sql_{dialect}_list"].apply(
    lambda x: ";".join(x) + ";"
)

merged_df.fillna("", inplace=True)

# change all db_type to dialect
merged_df["db_type"] = dialect

# drop original query col and table_metadata_string col
merged_df.drop(columns=["query", "query_list", "table_metadata_string"], inplace=True)

# rename sql_{dialect} to sql
merged_df.rename(
    columns={
        f"sql_{dialect}_list": "query",
    },
    inplace=True,
)

# drop cols
drop_columns = [
    "valid_list",
    "err_msg_list",
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

# save to csv
merged_df.to_csv(output_file, index=False)
print(f"Saved to {output_file}")
print(
    """\n\nNote that translations may not be 100% accurate and may require manual correction, especially for date-related syntax such as the following:
date arithmetic calculations, date interval functions, date truncations, date part extractions, current date/time functions
Do also check that all SQL syntax in instructions are correctly translated. 
Instruction translation in `instructions_to_<dialect>` of utils/dialects.py is not performed by an LLM and currently only handle specific cases."""
)
