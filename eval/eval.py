# this file contains all of the helper functions used for evaluations

import re
from func_timeout import func_timeout
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from sqlalchemy import create_engine

# like_pattern = r"LIKE\s+'[^']*'"
like_pattern = r"LIKE[\s\S]*'"


def normalize_table(
    df: pd.DataFrame, query_category: str, question: str
) -> pd.DataFrame:
    """
    Normalizes a dataframe by:
    1. removing all duplicate rows
    2. sorting columns in alphabetical order
    3. sorting rows using values from first column to last (if query_category is not 'order_by' and question does not ask for ordering)
    4. resetting index
    """
    # remove duplicate rows, if any
    df = df.drop_duplicates()

    # sort columns in alphabetical order
    sorted_df = df.reindex(sorted(df.columns), axis=1)

    # check if query_category is 'order_by' and if question asks for ordering
    has_order_by = False
    pattern = re.compile(r"(order|sort|arrange)", re.IGNORECASE)
    in_question = re.search(pattern, question.lower())  # true if contains
    if query_category == "order_by" or in_question:
        has_order_by = True
    if not has_order_by:
        # sort rows using values from first column to last
        sorted_df = sorted_df.sort_values(by=list(sorted_df.columns))
    # reset index
    sorted_df = sorted_df.reset_index(drop=True)
    return sorted_df


# for escaping percent signs in regex matches
def escape_percent(match):
    # Extract the matched group
    group = match.group(0)
    # Replace '%' with '%%' within the matched group
    escaped_group = group.replace("%", "%%")
    # Return the escaped group
    return escaped_group


def query_postgres_db(
    query: str, db_name: str, db_creds: dict = None, timeout: float = 10.0
) -> pd.DataFrame:
    """
    Runs query on postgres db and returns results as a dataframe.
    This assumes that you have the evaluation database running locally.
    If you don't, you can following the instructions in the README (Restoring to Postgres) to set it up.

    timeout: time in seconds to wait for query to finish before timing out
    """
    if db_creds is None:
        db_creds = {
            "host": "localhost",
            "port": 5432,
            "user": "postgres",
            "password": "postgres",
            "database": db_name,
        }
    try:
        db_url = f"postgresql://{db_creds['user']}:{db_creds['password']}@{db_creds['host']}:{db_creds['port']}/{db_name}"
        engine = create_engine(db_url)
        escaped_query = re.sub(
            like_pattern, escape_percent, query, flags=re.IGNORECASE
        )  # ignore case of LIKE
        results_df = func_timeout(
            timeout, pd.read_sql_query, args=(escaped_query, engine)
        )
        engine.dispose()  # close connection
        return results_df
    except Exception as e:
        if engine:
            engine.dispose()  # close connection if query fails/timeouts
        raise e


def compare_df(
    df1: pd.DataFrame, df2: pd.DataFrame, query_category: str, question: str
) -> bool:
    """
    Compares two dataframes and returns True if they are the same, else False.
    """
    # drop duplicates to ensure equivalence
    df1 = df1
    df2 = df2
    if df1.shape == df2.shape and (df1.values == df2.values).all():
        return True

    df1 = normalize_table(df1, query_category, question)
    df2 = normalize_table(df2, query_category, question)
    # try:
    #     assert_frame_equal(df1, df2, check_dtype=False, check_names=False)  # handles dtype mismatches
    # except AssertionError:
    #     return False
    if df1.shape == df2.shape and (df1.values == df2.values).all():
        return True
    else:
        return False


def subset_df(
    df_sub: pd.DataFrame,
    df_super: pd.DataFrame,
    query_category: str,
    question: str,
    verbose: bool = False,
) -> bool:
    """
    Checks if df_sub is a subset of df_super
    """
    if df_sub.empty:
        return False  # handle cases for empty dataframes

    # make a copy of df_super so we don't modify the original while keeping track of matches
    df_super_temp = df_super.copy(deep=True)
    matched_columns = []
    for col_sub_name in df_sub.columns:
        col_match = False
        for col_super_name in df_super_temp.columns:
            col_sub = df_sub[col_sub_name].sort_values().reset_index(drop=True)
            col_super = (
                df_super_temp[col_super_name].sort_values().reset_index(drop=True)
            )
            try:
                assert_series_equal(
                    col_sub, col_super, check_dtype=False, check_names=False
                )
                col_match = True
                matched_columns.append(col_super_name)
                # remove col_super_name to prevent us from matching it again
                df_super_temp = df_super_temp.drop(columns=[col_super_name])
                break
            except AssertionError:
                continue
        if col_match == False:
            if verbose:
                print(f"no match for {col_sub_name}")
            return False
    df_sub_normalized = normalize_table(df_sub, query_category, question)

    # get matched columns from df_super, and rename them with columns from df_sub, then normalize
    df_super_matched = df_super[matched_columns].rename(
        columns=dict(zip(matched_columns, df_sub.columns))
    )
    df_super_matched = normalize_table(df_super_matched, query_category, question)

    try:
        assert_frame_equal(df_sub_normalized, df_super_matched, check_dtype=False)
        return True
    except AssertionError:
        return False
