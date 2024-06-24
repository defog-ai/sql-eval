<|start_header_id|>user<|end_header_id|>

Generate a SQL query to answer this question: `{user_question}`
{instructions}
DDL statements:
{table_metadata_string}

{cot_instructions}Generate a valid SQL query that best answers the question `{user_question}`.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

I will reflect on the user's request before answering the question.

The question for which a SQL query must be generated is the following: `{user_question}`

{instruction_reflections}
With this in mind, here is the SQL query that best answers the question `{user_question}`, and only references the appropriate tables and columns in the DDL statements:
```sql
