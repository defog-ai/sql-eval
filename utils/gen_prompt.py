from utils.pruning import prune_metadata_str, to_prompt_schema
import os


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
            if public_data:
                import defog_data.supplementary as sup

                column_join = sup.columns_join.get(db_name, [])
            else:
                import defog_data_private.supplementary as sup

                column_join = sup.columns_join.get(db_name, [])

            join_list = []
            for values in column_join.values():
                col_1, col_2 = values[0]
                # add to join_list
                join_str = f"{col_1} can be joined with {col_2}"
                if join_str not in join_list:
                    join_list.append(join_str)

            if len(join_list) > 0:
                join_list = "\n\n- " + "\n- ".join(join_list)

            md = dbs[db_name]["table_metadata"]
            table_metadata_string = to_prompt_schema(md, shuffle_metadata) + join_list
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
