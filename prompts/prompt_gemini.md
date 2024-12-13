Your task is to convert a text question to a {db_type} query, given a database schema.

Generate a SQL query that answers the question `{user_question}`.
{instructions}
This query will run on a database whose schema is represented in this SQL DDL:
{table_metadata_string}

Return the SQL query that answers the question `{user_question}`
```sql