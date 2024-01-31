# this file contains all of the helper functions used for evaluations

import itertools
import re
from func_timeout import func_timeout
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from sqlalchemy import create_engine
from utils.creds import db_creds_all

LIKE_PATTERN = r"LIKE[\s\S]*'"


def normalize_table(
    df: pd.DataFrame, query_category: str, question: str, sql: str = None
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

    # sort columns in alphabetical order of column names
    sorted_df = df.reindex(sorted(df.columns), axis=1)

    # check if query_category is 'order_by' and if question asks for ordering
    has_order_by = False
    pattern = re.compile(r"\b(order|sort|arrange)\b", re.IGNORECASE)
    in_question = re.search(pattern, question.lower())  # true if contains
    if query_category == "order_by" or in_question:
        has_order_by = True

        if sql:
            # determine which columns are in the ORDER BY clause of the sql generated, using regex
            pattern = re.compile(r"ORDER BY[\s\S]*", re.IGNORECASE)
            order_by_clause = re.search(pattern, sql)
            if order_by_clause:
                order_by_clause = order_by_clause.group(0)
                # get all columns in the ORDER BY clause, by looking at the text between ORDER BY and the next semicolon, comma, or parantheses
                pattern = re.compile(r"(?<=ORDER BY)(.*?)(?=;|,|\))", re.IGNORECASE)
                order_by_columns = re.findall(pattern, order_by_clause)[0].split()
                order_by_columns = [
                    col.strip().split(".")[-1] for col in order_by_columns
                ]

                ascending = False
                # if there is a DESC or ASC in the ORDER BY clause, set the ascending to that
                if "DESC" in [i.upper() for i in order_by_columns]:
                    ascending = False
                elif "ASC" in [i.upper() for i in order_by_columns]:
                    ascending = True

                # remove whitespace, commas, and parantheses
                order_by_columns = [col.strip() for col in order_by_columns]
                order_by_columns = [
                    col.replace(",", "").replace("(", "") for col in order_by_columns
                ]
                order_by_columns = [
                    i
                    for i in order_by_columns
                    if i.lower()
                    not in ["desc", "asc", "nulls", "last", "first", "limit"]
                ]

                # get all columns in sorted_df that are not in order_by_columns
                other_columns = [
                    i for i in sorted_df.columns.tolist() if i not in order_by_columns
                ]

                # only choose order_by_columns that are in sorted_df
                order_by_columns = [
                    i for i in order_by_columns if i in sorted_df.columns.tolist()
                ]
                sorted_df = sorted_df.sort_values(
                    by=order_by_columns + other_columns, ascending=ascending
                )

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


# find start and end index of { } in a string. return (start, end) if found, else return (-1, -1)
def find_bracket_indices(s: str, start_index: int = 0) -> "tuple[int, int]":
    start = s.find("{", start_index)
    end = s.find("}", start + 1)
    if start == -1 or end == -1:
        return (-1, -1)
    return (start, end)


# extrapolate all possible queries from a query with { } in it
def get_all_minimal_queries(query: str) -> "list[str]":
    """
    extrapolate all possible queries
    - split by semicolon. this is to accommodate queries where joins to other tables are also acceptable.
    - expand all column permutations if there are braces { } in it. eg:
    ```sql
        SELECT {user.id, user.name} FROM user;
    ```
    Would be expanded to:
    ```sql
        SELECT user.id FROM user;
        SELECT user.name FROM user;
        SELECT user.id, user.name FROM user;
    ```
    """
    queries = query.split(";")
    result_queries = []
    for query in queries:
        query = query.strip()
        if query == "":
            continue
        start, end = find_bracket_indices(query, 0)
        if (start, end) == (-1, -1):
            result_queries.append(query)
            continue
        else:
            # get all possible column subsets
            column_options = query[start + 1 : end].split(",")
            column_combinations = list(
                itertools.chain.from_iterable(
                    itertools.combinations(column_options, r)
                    for r in range(1, len(column_options) + 1)
                )
            )
            for column_tuple in column_combinations:
                left = query[:start]
                column_str = ", ".join(column_tuple)
                right = query[end + 1 :]
                # change group by size dynamically if necessary
                if right.find("GROUP BY {}"):
                    right = right.replace("GROUP BY {}", f"GROUP BY {column_str}")
                result_queries.append(left + column_str + right)
    return result_queries


def query_postgres_db(
    query: str, db_name: str, db_creds: dict = None, timeout: float = 10.0
) -> pd.DataFrame:
    """
    Runs query on postgres db and returns results as a dataframe.
    This assumes that you have the evaluation database running locally.
    If you don't, you can following the instructions in the README (Start Postgres Instance) to set it up.

    timeout: time in seconds to wait for query to finish before timing out
    """
    if db_creds is None:
        db_creds = db_creds_all["postgres"]
    try:
        db_url = f"postgresql://{db_creds['user']}:{db_creds['password']}@{db_creds['host']}:{db_creds['port']}/{db_name}"
        engine = create_engine(db_url)
        escaped_query = re.sub(
            LIKE_PATTERN, escape_percent, query, flags=re.IGNORECASE
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


def query_snowflake_db(
    query: str, db_name: str, db_creds: dict = None, timeout: float = 10.0
) -> pd.DataFrame:
    """
    Runs query on snowflake db and returns results as a dataframe.
    This assumes that you have the evaluation database set up on Snowflake.
    If you don't, you can following the instructions in the README (Snowflake Setup) to set it up.

    timeout: time in seconds to wait for query to finish before timing out
    """

    import snowflake.connector

    if db_creds is None:
        db_creds = db_creds_all["snowflake"]

    try:
        conn = snowflake.connector.connect(
            user=db_creds["user"],
            password=db_creds["password"],
            account=db_creds["account"],
        )
        cur = conn.cursor()
        cur.execute(f"USE WAREHOUSE {db_creds['warehouse']}")  # set the warehouse
        cur.execute(f"USE DATABASE {db_name}")  # set the database
        cur.execute(query)
        colnames = [desc[0] for desc in cur.description]
        results = cur.fetchall()
        cur.close()
        conn.close()
        # make into a dataframe
        df = pd.DataFrame(results, columns=colnames)
        return df
    except Exception as e:
        raise e


def compare_df(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    query_category: str,
    question: str,
    query_gold: str = None,
    query_gen: str = None,
) -> bool:
    """
    Compares two dataframes and returns True if they are the same, else False.
    """
    # drop duplicates to ensure equivalence
    if df1.shape == df2.shape and (df1.values == df2.values).all():
        return True

    df1 = normalize_table(df1, query_category, question, query_gold)
    df2 = normalize_table(df2, query_category, question, query_gen)

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
    df_sub_normalized = normalize_table(df_sub, query_category, question, "")

    # get matched columns from df_super, and rename them with columns from df_sub, then normalize
    df_super_matched = df_super[matched_columns].rename(
        columns=dict(zip(matched_columns, df_sub.columns))
    )
    df_super_matched = normalize_table(df_super_matched, query_category, question, "")

    try:
        assert_frame_equal(df_sub_normalized, df_super_matched, check_dtype=False)
        return True
    except AssertionError:
        return False


def compare_query_results(
    query_gold: str,
    query_gen: str,
    db_name: str,
    db_type: str,
    db_creds: dict,
    question: str,
    query_category: str,
    timeout: float = 10.0,
) -> "tuple[bool, bool]":
    """
    Compares the results of two queries and returns a tuple of booleans, where the first element is
    whether the queries produce exactly the same result, and the second element is whether the
    result of the gold query is a subset of the result of the generated query (still correct).
    We bubble up exceptions (mostly from query_postgres_db) to be handled in the runner.
    """
    queries_gold = get_all_minimal_queries(query_gold)
    if db_type == "postgres":
        results_gen = query_postgres_db(query_gen, db_name, db_creds, timeout)
    elif db_type == "snowflake":
        results_gen = query_snowflake_db(query_gen, db_name, db_creds, timeout)
    else:
        raise ValueError(
            f"Invalid db_type: {db_type}. Only postgres and snowflake are supported."
        )
    correct = False
    for q in queries_gold:
        if db_type == "postgres":
            results_gold = query_postgres_db(q, db_name, db_creds, timeout)
        elif db_type == "snowflake":
            results_gold = query_snowflake_db(q, db_name, db_creds, timeout)
        else:
            raise ValueError(
                f"Invalid db_type: {db_type}. Only postgres and snowflake are supported."
            )
        if compare_df(
            results_gold, results_gen, query_category, question, query_gold, query_gen
        ):
            return (True, True)
        elif subset_df(results_gold, results_gen, query_category, question):
            correct = True
    return (False, correct)
