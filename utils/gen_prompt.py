from utils.pruning import prune_metadata_str, to_prompt_schema


def generate_prompt(
    prompt_file,
    question,
    db_name,
    instructions="",
    k_shot_prompt="",
    glossary="",
    table_metadata_string="",
    prev_invalid_sql="",
    prev_error_msg="",
    question_0="",
    query_0="",
    question_1="",
    query_1="",
    public_data=True,
    columns_to_keep=40,
    shuffle_metadata=False,
):

    from defog_data.metadata import dbs  # to avoid CI error

    with open(prompt_file, "r") as f:
        prompt = f.read()
    question_instructions = question + " " + instructions

    if table_metadata_string == "":
        if columns_to_keep > 0:
            table_metadata_string = prune_metadata_str(
                question_instructions,
                db_name,
                public_data,
                columns_to_keep,
                shuffle_metadata,
            )
        elif columns_to_keep == 0:
            md = dbs[db_name]["table_metadata"]
            table_metadata_string = to_prompt_schema(md, shuffle_metadata)
        else:
            raise ValueError("columns_to_keep must be >= 0")
    if glossary == "":
        glossary = dbs[db_name]["glossary"]

    prompt = prompt.format(
        user_question=question,
        instructions=instructions,
        table_metadata_string=table_metadata_string,
        k_shot_prompt=k_shot_prompt,
        glossary=glossary,
        prev_invalid_sql=prev_invalid_sql,
        prev_error_msg=prev_error_msg,
        question_0=question_0,
        query_0=query_0,
        question_1=question_1,
        query_1=query_1,
    )
    return prompt
