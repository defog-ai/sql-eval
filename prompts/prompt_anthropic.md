Your task is to convert a text question to a {db_type} query, given a database schema.

The question that you must generate a SQL for is this `{user_question}`.
{instructions}
This query will run on a database with the following schema:
{table_metadata_string}
{k_shot_prompt}

Just return the SQL query, nothing else.