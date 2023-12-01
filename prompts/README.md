# Defining your prompt
You can define your prompt template by using the following variables:
- `user_question`: The question that we want to generate sql for
- `table_metadata_string`: The metadata of the table that we want to query. This is a string that contains the table names, column names and column types. This allows the model to know which columns/tables are available for getting information from. For the sqlcoder model that we released, you would need to represent your table metadata as a [SQL DDL](https://en.wikipedia.org/wiki/Data_definition_language) statement.
- `instructions`: This is an optional field that allows you to customize specific instructions for each question, if needed. For example, if you want to ask the model to format your dates a particular way, define keywords, or adapt the SQL to a different database, you can do so here. If you don't need to customize the instructions, you can omit this in your prompt.

Here is how a sample might look like with the above variables:
```markdown
### Task
Generate a SQL query to answer the following question:
`{user_question}`
`{instructions}`
### Database Schema
The query will run on a database with the following schema:
{table_metadata_string}

### Answer
Given the database schema, here is the SQL query that answers `{user_question}`:
```sql
```

# Adding variables
You can add variables using curly braces - like so `{user_question}`. These can then be updated at runtime using Python's `.format()` method for strings. Like [here](../eval/hf_runner.py#L18).

# Translating to OpenAI's messages prompt
If you're performing evaluation with OpenAI's chat models, please ensure that your prompt contains the keywords `### Instructions:`, `### Input:`, and `### Response:`. This will help ensure that the prompt sections are automatically mapped to OpenAI's different prompt roles. The text under Instructions, Input and Response will be converted to the `system`, `user` and `assistant` prompts respectively.