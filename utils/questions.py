from typing import Optional
import pandas as pd


def prepare_questions_df(
    questions_file: str,
    db_type: str,
    num_questions: Optional[int] = None,
    k_shot: bool = False,
):
    question_query_df = pd.read_csv(questions_file, nrows=num_questions)
    question_query_df["db_type"] = db_type
    question_query_df["generated_query"] = ""
    question_query_df["reason"] = ""
    question_query_df["error_msg"] = ""
    question_query_df["exact_match"] = 0
    question_query_df["correct"] = 0
    question_query_df["error_query_gen"] = 0
    question_query_df["error_db_exec"] = 0
    question_query_df["timeout"] = 0
    # add custom metrics below:
    question_query_df["latency_seconds"] = 0.0  # latency of query generation in seconds
    question_query_df["tokens_used"] = 0  # number of tokens used in query generation

    # get instructions if applicable
    if "instructions" in question_query_df.columns:
        question_query_df["instructions"] = question_query_df["instructions"].fillna("")
        question_query_df["instructions"] = question_query_df["instructions"].apply(
            lambda x: x.replace(". ", ".\n")
        )
        question_query_df["instructions"] = question_query_df["instructions"].apply(
            lambda x: f"Instructions:\n{x}\n"
        )
    else:
        question_query_df["instructions"] = ""

    # get k_shot prompt if applicable
    if not k_shot:
        question_query_df["k_shot_prompt"] = ""
    else:
        if "k_shot_prompt" not in question_query_df.columns:
            raise ValueError(
                "k_shot is True but k_shot_prompt column not in questions file"
            )
        else:
            question_query_df["k_shot_prompt"] = question_query_df[
                "k_shot_prompt"
            ].fillna("")
            question_query_df["k_shot_prompt"] = question_query_df[
                "k_shot_prompt"
            ].apply(lambda x: x.replace("\\n", "\n"))
            question_query_df["k_shot_prompt"] = question_query_df[
                "k_shot_prompt"
            ].apply(
                lambda x: f"\nAdhere closely to the following correct examples as references for answering the question:\n{x}"
            )

    question_query_df.reset_index(inplace=True, drop=True)
    return question_query_df
