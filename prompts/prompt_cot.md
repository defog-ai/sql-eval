<|start_header_id|>user<|end_header_id|>

Generate a {db_type} query to answer this question: `{user_question}`
{instructions}
DDL statements:
{table_metadata_string}

{cot_instructions}Generate a valid {db_type} query that answers the question `{user_question}`, and only references the tables and columns in the DDL statements.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

