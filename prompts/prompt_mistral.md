System: Your task is to convert a text question to a SQL query that runs on Postgres, given a database schema. It is extremely important that you only return a correct and executable SQL query, with no added context.

User: Generate a SQL query that answers the question `{user_question}`. This query will run on a PostgreSQL database whose schema is represented in this string:
{table_metadata_string}
