### Instructions:
Your task is to generate a correct SQL query for answering a specific question about a SQL database. While doing this, you must adhere to the following guidelines:

- Use exact tables and columns from database. Only reference the tables and columns that are explicitly listed in the provided database schema.
- Use Table Aliases to prevent any ambiguity. For example, `SELECT table1.column_name1, table2.column_name1 FROM table1 JOIN table2 ON table1.id = table2.id`
- Use the "ILIKE" operator with '%' wildcards when querying text columns, e.g., `column_name ILIKE '%search_text%'`.
- When creating a ratio, always cast the numerator as float, e.g., `CAST(numerator AS float) / denominator`.

### Input:
Create a SQL query that answers the question `{user_question}`.

The query will run on a database whose schema is represented in this string:
{table_metadata_string}

### Response:
Here is the SQL query I have generated to answer the question `{user_question}`:
```sql