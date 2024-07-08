<|start_header_id|>system<|end_header_id|>

Follow instructions to the letter, and answer questions without making any additional assumptions.<|start_header_id|>user<|end_header_id|>

Generate a {db_type} query to answer this question: `{user_question}`
{instructions}
DDL statements:
{table_metadata_string}

{cot_instructions}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

I was asked to generate a SQL query for this question: `{user_question}`

{instruction_reflections}
Here is the {db_type} query that best answers the question:
```sql
