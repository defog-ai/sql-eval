<|start_header_id|>system<|end_header_id|>

Follow instructions to the letter, and answer questions without making any additional assumptions.<|start_header_id|>user<|end_header_id|>

Generate a SQL query to answer this question: `{user_question}`
{instructions}
DDL statements:
{table_metadata_string}

{cot_instructions}Generate a valid SQL query that best answers the question `{user_question}`.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

I will reflect on the user's request before answering the question.

I was asked to generate a SQL query for this question: `{user_question}`

{instruction_reflections}
With this in mind, here is the SQL query that best answers the question while only using appropriate tables and columns from the DDL statements:
```sql
