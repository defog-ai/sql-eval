# Defining your prompt
You can define your prompt in the following structure way

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
You can add variables using curly braces - like so `{user_question}`. Then, these can be updated at runtime using Python's `.format()` function for strings. Like [here](../eval/hf_runner.py#L18)

# Translating to OpenAI's messages prompt
If evaluating OpenAI's chat models, please ensure that your prompt always has the keywords `### Instructions:`, `### Input:`, and `### Response:` in them. This will help ensure that the model is automatically converted to OpenAI's `system`, `user`, and `assistant` prompts. The section under Instructions is mapped to the `system` prompt, the section under Input is mapped to the `user` prompt, and the section under Response is mapped to the `assistant` prompt