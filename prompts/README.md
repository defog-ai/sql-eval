# Defining your prompt
You can define your prompt using the following structure.

```
### Instructions:
YOUR INSTRUCTIONS FOR THE MODEL

### Input:
Generate a SQL query that answers the question `{user_question}`.
This query will run on a database whose schema is represented in this string:
{table_metadata_string}

### Response:
THE RESPONSE TEXT FOR THE MODEL
```sql
```

# Adding variables
You can add variables using curly braces - like so `{user_question}`. These can then be updated at runtime using Python's `.format()` method for strings. Like [here](../eval/hf_runner.py#L18).

# Translating to OpenAI's messages prompt
If you're performing evaluation with OpenAI's chat models, please ensure that your prompt contains the keywords `### Instructions:`, `### Input:`, and `### Response:`. This will help ensure that the prompt sections are automatically mapped to OpenAI's different prompt roles. The text under Instructions, Input and Response will be converted to the `system`, `user` and `assistant` prompts respectively.