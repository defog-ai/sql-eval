from typing import Dict, List, Optional
import numpy as np
from utils.dialects import (
    ddl_to_bigquery,
    ddl_to_mysql,
    ddl_to_sqlite,
    ddl_to_tsql,
    get_schema_names,
)


def to_prompt_schema(
    md: Dict[str, List[Dict[str, str]]], seed: Optional[int] = None
) -> str:
    """
    Return a DDL statement for creating tables from a metadata dictionary
    `md` has the following structure:
        {'table1': [
            {'column_name': 'col1', 'data_type': 'int', 'column_description': 'primary key'},
            {'column_name': 'col2', 'data_type': 'text', 'column_description': 'not null'},
            {'column_name': 'col3', 'data_type': 'text', 'column_description': ''},
        ],
        'table2': [
        ...
        ]},
    This is just for converting the dictionary structure of one's metadata into a string
    for pasting into prompts, and not meant to be used to initialize a database.
    seed is used to shuffle the order of the tables when not None
    """
    md_create = ""
    table_names = list(md.keys())
    if seed:
        np.random.seed(seed)
        np.random.shuffle(table_names)
    for table in table_names:
        md_create += f"CREATE TABLE {table} (\n"
        columns = md[table]
        if seed:
            np.random.seed(seed)
            np.random.shuffle(columns)
        for i, column in enumerate(columns):
            col_name = column["column_name"]
            # if column name has spaces, wrap it in double quotes
            if " " in col_name:
                col_name = f'"{col_name}"'
            dtype = column["data_type"]
            col_desc = column.get("column_description", "").replace("\n", " ")
            if col_desc:
                col_desc = f" --{col_desc}"
            if i < len(columns) - 1:
                md_create += f"  {col_name} {dtype},{col_desc}\n"
            else:
                # avoid the trailing comma for the last line
                md_create += f"  {col_name} {dtype}{col_desc}\n"
        md_create += ");\n"
    return md_create


def generate_aliases(table_names: list) -> str:
    """
    Generate aliases for table names
    """
    aliases = {}
    reserved_keywords = [
        "all",
        "and",
        "any",
        "as",
        "asc",
        "do",
        "end",
        "for",
        "in",
        "is",
        "not",
        "to",
    ]
    for table_name in table_names:
        if "." in table_name:
            table_name = table_name.split(".", 1)[1]
        alias = table_name[0]
        if (
            alias in aliases.values() and "_" in table_name
        ) or alias.lower() in reserved_keywords:
            alias = table_name.split("_")[0][0] + table_name.split("_")[1][0]
        if alias in aliases.values() or alias.lower() in reserved_keywords:
            alias = table_name[:2]
        if alias in aliases.values() or alias.lower() in reserved_keywords:
            alias = table_name[:3]
        num = 2
        while alias in aliases.values() or alias.lower() in reserved_keywords:
            alias = table_name[0] + str(num)
            num += 1

        aliases[table_name] = alias

    aliases_str = ""
    for table_name, alias in aliases.items():
        aliases_str += f"-- {table_name} AS {alias}\n"
    return aliases_str


def generate_prompt(
    prompt_file,
    question,
    db_name,
    db_type="postgres",
    instructions="",
    k_shot_prompt="",
    glossary="",
    table_metadata_string="",
    prev_invalid_sql="",
    prev_error_msg="",
    question_0="",
    query_0="",
    question_1="",
    query_1="",
    cot_instructions="",
    cot_pregen=False,
    public_data=True,
    columns_to_keep=40,
    shuffle_metadata=False,
):
    from defog_data.metadata import dbs  # to avoid CI error

    with open(prompt_file, "r") as f:
        prompt = f.read()
    question_instructions = question + " " + instructions
    table_names = []

    join_str = ""
    # retrieve metadata, either pruned or full
    if table_metadata_string == "":
        if columns_to_keep > 0:
            from utils.pruning import prune_metadata_str

            table_metadata_ddl, join_str = prune_metadata_str(
                question_instructions,
                db_name,
                public_data,
                columns_to_keep,
                shuffle_metadata,
            )
            # remove triple backticks
            table_metadata_ddl = table_metadata_ddl.replace("```", "").strip()
        elif columns_to_keep == 0:
            if public_data:
                import defog_data.supplementary as sup

                column_join = sup.columns_join.get(db_name, {})
            else:
                import defog_data_private.supplementary as sup

                column_join = sup.columns_join.get(db_name, {})

            md = dbs[db_name]["table_metadata"]
            table_names = list(md.keys())
            table_metadata_ddl = to_prompt_schema(md, shuffle_metadata)

            # get join_str from column_join
            join_list = []
            for values in column_join.values():
                col_1, col_2 = values[0]
                # add to join_list
                join_str = f"{col_1} can be joined with {col_2}"
                if join_str not in join_list:
                    join_list.append(join_str)
            if len(join_list) > 0:
                join_str = "\nHere is a list of joinable columns:\n" + "\n".join(
                    join_list
                )
            else:
                join_str = ""
        else:
            raise ValueError("columns_to_keep must be >= 0")

        # add schema creation statements if relevant
        schema_names = get_schema_names(table_metadata_ddl)
        if schema_names:
            for schema_name in schema_names:
                table_metadata_ddl = (
                    f"CREATE SCHEMA IF NOT EXISTS {schema_name};\n" + table_metadata_ddl
                )
        # remove schema names from cot_instructions if db_type is in ["sqlite", "mysql", "bigquery"]
        if db_type in ["sqlite", "mysql", "bigquery"]:
            for schema_name in schema_names:
                cot_instructions = cot_instructions.replace(f"{schema_name}.", "")

        # transform metadata string to target dialect if necessary
        if db_type in ["postgres", "snowflake"]:
            table_metadata_string = table_metadata_ddl
        elif db_type == "bigquery":
            table_metadata_string = ddl_to_bigquery(
                table_metadata_ddl, "postgres", db_name, ""
            )[0]
        elif db_type == "mysql":
            table_metadata_string = ddl_to_mysql(
                table_metadata_ddl, "postgres", db_name, ""
            )[0]
        elif db_type == "sqlite":
            table_metadata_string = ddl_to_sqlite(
                table_metadata_ddl, "postgres", db_name, ""
            )[0]
        elif db_type == "tsql":
            table_metadata_string = ddl_to_tsql(
                table_metadata_ddl, "postgres", db_name, ""
            )[0]
        else:
            raise ValueError(
                "db_type must be one of postgres, snowflake, bigquery, mysql, sqlite, or tsql"
            )
    if glossary == "":
        glossary = dbs[db_name]["glossary"]

    instruction_reflections = instructions.replace(
        "\nFollow the instructions below to generate the query:",
        "\nAdditionally, I was asked to follow the instructions below to generate the query:",
    )
    instruction_reflections = instruction_reflections + "\n"

    prompt = prompt.format(
        user_question=question,
        db_type=db_type,
        instructions=instructions,
        table_metadata_string=table_metadata_string,
        k_shot_prompt=k_shot_prompt,
        glossary=glossary,
        prev_invalid_sql=prev_invalid_sql,
        prev_error_msg=prev_error_msg,
        question_0=question_0,
        query_0=query_0,
        question_1=question_1,
        query_1=query_1,
        cot_instructions=cot_instructions,
        instruction_reflections=instruction_reflections,
        join_hints=join_str,
    )

    if cot_pregen:
        table_aliases = generate_aliases(table_names)
        prompt = prompt + table_aliases
    return prompt
