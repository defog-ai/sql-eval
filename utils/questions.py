from typing import Optional
import pandas as pd


def prepare_questions_df(questions_file: str, num_questions: Optional[int] = None):
    question_query_df = pd.read_csv(questions_file, nrows=num_questions)
    # optional instructions:
    if "instructions" in question_query_df.columns:
        question_query_df["instructions"] = question_query_df["instructions"].fillna("")
    else:
        question_query_df["instructions"] = ""
    # add columns for standard metrics
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

    question_query_df.reset_index(inplace=True, drop=True)
    return question_query_df
