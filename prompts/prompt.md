### Instructions:
Your task is to convert a text question to a SQL query that runs on Postgres, given a database schema.

Adhere to the following guidelines for generating the query:
- **Carefully go through the question and database schema word by word**, and create a query that most appropriately answers the question
- **Use Table Aliases** to prevent any ambiguity. For example, `SELECT table1.col1, table2.col1 FROM table1 JOIN table2 ON table1.id = table2.id`.
- When creating a ratio, always cast the numerator as float

### Input:
Generate a SQL query that answers the question `{user_question}`.

This query will run on a database whose schema is represented in this string:
{table_metadata_string}

### Generated SQL:
Based on your instructions, here is the SQL query I have generated to answer the question `{user_question}`:
```sql