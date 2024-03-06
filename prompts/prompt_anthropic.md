### Instructions:
Your task is to convert a text question to a SQL query that runs on Postgres, given a database schema. Return the SQL as a markdown string, nothing else.

### Input:
Generate a SQL query that answers the question `{user_question}`.
{instructions}{glossary}
This query will run on a database whose schema is represented in this string:
{table_metadata_string}
{k_shot_prompt}
### Response:
Given the database schema, here is the SQL query that answers `{user_question}`:
```sql
