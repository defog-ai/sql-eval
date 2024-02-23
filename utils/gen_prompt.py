from utils.pruning import prune_metadata_str


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
    public_data=True,
):
    with open(prompt_file, "r") as f:
        prompt = f.read()
    question_instructions = question + " " + instructions

    if table_metadata_string == "":
        pruned_metadata_str = prune_metadata_str(
            question_instructions, db_name, public_data
        )
    else:
        pruned_metadata_str = table_metadata_string

    prompt = prompt.format(
        user_question=question,
        instructions=instructions,
        table_metadata_string=pruned_metadata_str,
        k_shot_prompt=k_shot_prompt,
        glossary=glossary,
        prev_invalid_sql=prev_invalid_sql,
        prev_error_msg=prev_error_msg,
    )
    return prompt
