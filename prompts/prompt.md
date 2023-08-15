### Instructions:
Your task is to convert a text question to a SQL query that runs on a PostgreSQL database, given a database schema.

You must adhere to these rules:
- **Deliberately go through the question and database schema word by word**, and create a query that most appropriately answers the question
- **Use Table Aliases** to prevent ambiguity. For example, `SELECT table1.col1, table2.col1 FROM table1 INNER JOIN table2 ON table1.id = table2.id`.
- When creating a ratio, always cast the numerator as float

### Input:
Generate a SQL query that answers the question `{user_question}`.

This query will run on a database whose schema is represented in this string:
{table_metadata_string}

### Response:
Based on your instructions, here is a valid SQL query I have generated to answer the question `{user_question}`:
```sql