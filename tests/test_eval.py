from eval.eval import (
    compare_query_results,
    find_bracket_indices,
    get_all_minimal_queries,
    normalize_table,
    compare_df,
    query_postgres_db,
    query_snowflake_db,
    subset_df,
)
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from unittest import mock

query = "SELECT * FROM table_name"
query_order_by = "SELECT * FROM table_name ORDER BY name DESC"


@pytest.fixture
def unordered_dataframe():
    # Create a sample DataFrame for testing
    data = {
        "name": ["John", "Alice", "Jane"],
        "age": [25, 35, 30],
        "city": ["New York", "Paris", "London"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def test_dataframes():
    df0 = pd.DataFrame({"A": [], "B": []})
    df0_same = pd.DataFrame({"A": [], "B": []})
    df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    df2 = pd.DataFrame({"A": [1, 2, 3, 4], "B": [4, 5, 5, 6]})
    df1_same = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    df1_value_diff = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 7]})
    df1_columns_reordered = pd.DataFrame({"B": [4, 5, 6], "A": [1, 2, 3]})
    df1_columns_diffcase = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df1_columns_renamed = pd.DataFrame({"C": [1, 2, 3], "D": [4, 5, 6]})
    df1_rows_reordered = pd.DataFrame({"A": [2, 1, 3], "B": [5, 4, 6]})
    df1_rows_reordered_columns_renamed = pd.DataFrame({"X": [2, 1, 3], "Y": [5, 4, 6]})
    df1_rows_reordered_more_cols = pd.DataFrame(
        {"X": [2, 1, 3], "Y": [5, 4, 6], "Z": [7, 8, 9]}
    )
    df2_rows_reordered = pd.DataFrame({"A": [2, 1, 4, 3], "B": [5, 4, 6, 5]})
    df2_rows_reordered_sortb = pd.DataFrame({"A": [1, 3, 2, 4], "B": [4, 5, 5, 6]})
    df2_rows_reordered_sortb_renamed = pd.DataFrame(
        {"X": [1, 3, 2, 4], "Y": [4, 5, 5, 6]}
    )
    df2_rows_reordered_sortb_more_cols = pd.DataFrame(
        {"X": [1, 3, 2, 4], "Y": [4, 5, 5, 6], "Z": ["e", "b", "c", "d"]}
    )

    return (
        df0,
        df0_same,
        df1,
        df2,
        df1_same,
        df1_value_diff,
        df1_columns_reordered,
        df1_columns_diffcase,
        df1_columns_renamed,
        df1_rows_reordered,
        df1_rows_reordered_columns_renamed,
        df1_rows_reordered_more_cols,
        df2_rows_reordered,
        df2_rows_reordered_sortb,
        df2_rows_reordered_sortb_renamed,
        df2_rows_reordered_sortb_more_cols,
    )


def test_normalize_table(unordered_dataframe):
    # Test normalization without an order by clause: should sort columns and rows
    expected_df = pd.DataFrame(
        {
            "age": [25, 30, 35],
            "city": ["New York", "London", "Paris"],
            "name": ["John", "Jane", "Alice"],
        }
    )
    question = "What are the ages of the people in the table?"
    sql_query = "SELECT * FROM table_name"
    normalized_df = normalize_table(unordered_dataframe, query, question, sql_query)
    assert_frame_equal(expected_df, normalized_df)

    # Test normalization with an asc order by clause: should sort columns and rows (names ascending)
    expected_df = pd.DataFrame(
        {
            "age": [35, 30, 25],
            "city": ["Paris", "London", "New York"],
            "name": ["Alice", "Jane", "John"],  # names ascending
        }
    )
    question_sort_asc = "What are the ages of the people in the table? Sort by name."
    sql_query_sort_asc = "SELECT * FROM table_name ORDER BY name ASC"
    normalized_df = normalize_table(
        unordered_dataframe, "order_by", question_sort_asc, sql_query_sort_asc
    )
    assert_frame_equal(expected_df, normalized_df)

    # Test normalization with a desc order by clause: should sort columns and rows (names descending)
    expected_df = pd.DataFrame(
        {
            "age": [25, 30, 35],
            "city": ["New York", "London", "Paris"],
            "name": ["John", "Jane", "Alice"],  # names descending
        }
    )
    question_sort_desc = "What are the ages of the people in the table? Arrange by name starting from the largest."
    sql_query_sort_desc = "SELECT * FROM table_name ORDER BY name DESC"
    normalized_df = normalize_table(
        unordered_dataframe, "order_by", question_sort_desc, sql_query_sort_desc
    )
    assert_frame_equal(expected_df, normalized_df)


def test_find_bracket_indices():
    assert find_bracket_indices("hello {world} foo") == (6, 12)
    assert find_bracket_indices("hello {world} {foo}") == (6, 12)
    assert find_bracket_indices("hello {world") == (-1, -1)
    assert find_bracket_indices("hello world}") == (-1, -1)
    assert find_bracket_indices("hello world") == (-1, -1)


def test_get_all_minimal_queries():
    query1 = "SELECT * FROM persons WHERE persons.age > 25"
    assert get_all_minimal_queries(query1) == [query1]
    query2 = "SELECT persons.name FROM persons WHERE persons.age > 25 GROUP BY 1"
    assert get_all_minimal_queries(query2) == [query2]
    query3 = "SELECT {persons.name,persons.id} FROM persons WHERE persons.age > 25"
    option1 = "SELECT persons.name FROM persons WHERE persons.age > 25"
    option2 = "SELECT persons.id FROM persons WHERE persons.age > 25"
    option3 = "SELECT persons.name, persons.id FROM persons WHERE persons.age > 25"
    assert get_all_minimal_queries(query3) == [option1, option2, option3]
    query4 = "SELECT {persons.name,persons.id} FROM persons WHERE persons.age > 25 GROUP BY {}"
    option1 = (
        "SELECT persons.name FROM persons WHERE persons.age > 25 GROUP BY persons.name"
    )
    option2 = (
        "SELECT persons.id FROM persons WHERE persons.age > 25 GROUP BY persons.id"
    )
    option3 = "SELECT persons.name, persons.id FROM persons WHERE persons.age > 25 GROUP BY persons.name, persons.id"
    assert get_all_minimal_queries(query4) == [option1, option2, option3]
    query5 = "SELECT {persons.name,persons.id} FROM persons WHERE persons.age > 25 GROUP BY {};SELECT {user.name,user.id} FROM user WHERE user.age > 25 GROUP BY {};"
    option4 = "SELECT user.name FROM user WHERE user.age > 25 GROUP BY user.name"
    option5 = "SELECT user.id FROM user WHERE user.age > 25 GROUP BY user.id"
    option6 = "SELECT user.name, user.id FROM user WHERE user.age > 25 GROUP BY user.name, user.id"
    assert get_all_minimal_queries(query5) == [
        option1,
        option2,
        option3,
        option4,
        option5,
        option6,
    ]
    query6 = """WITH merchant_revenue AS (SELECT {m.mid,m.name}, m.category AS merchant_category, SUM(w.amount) AS total_revenue
    FROM consumer_div.merchants m
    INNER JOIN consumer_div.wallet_transactions_daily w ON m.mid = w.receiver_id AND w.receiver_type = 1
    WHERE w.status = 'success'
    GROUP BY {}, m.category)
    SELECT *, RANK() OVER (ORDER BY total_revenue DESC) AS mrr FROM merchant_revenue"""
    option7 = """WITH merchant_revenue AS (SELECT m.mid, m.category AS merchant_category, SUM(w.amount) AS total_revenue
    FROM consumer_div.merchants m
    INNER JOIN consumer_div.wallet_transactions_daily w ON m.mid = w.receiver_id AND w.receiver_type = 1
    WHERE w.status = 'success'
    GROUP BY m.mid, m.category)
    SELECT *, RANK() OVER (ORDER BY total_revenue DESC) AS mrr FROM merchant_revenue"""
    option8 = """WITH merchant_revenue AS (SELECT m.name, m.category AS merchant_category, SUM(w.amount) AS total_revenue
    FROM consumer_div.merchants m
    INNER JOIN consumer_div.wallet_transactions_daily w ON m.mid = w.receiver_id AND w.receiver_type = 1
    WHERE w.status = 'success'
    GROUP BY m.name, m.category)
    SELECT *, RANK() OVER (ORDER BY total_revenue DESC) AS mrr FROM merchant_revenue"""
    option9 = """WITH merchant_revenue AS (SELECT m.mid, m.name, m.category AS merchant_category, SUM(w.amount) AS total_revenue
    FROM consumer_div.merchants m
    INNER JOIN consumer_div.wallet_transactions_daily w ON m.mid = w.receiver_id AND w.receiver_type = 1
    WHERE w.status = 'success'
    GROUP BY m.mid, m.name, m.category)
    SELECT *, RANK() OVER (ORDER BY total_revenue DESC) AS mrr FROM merchant_revenue"""
    for expected, result in zip(
        get_all_minimal_queries(query6), [option7, option8, option9]
    ):
        assert expected == result
    assert get_all_minimal_queries(query6) == [option7, option8, option9]


@mock.patch("pandas.read_sql_query")
def test_query_postgres_db(mock_pd_read_sql_query):
    # note that we don't need to mock create_engine as it only contains the config,
    # but doesn't create the connection to the underlying db
    db_name = "test_db"
    db_creds = {
        "host": "localhost",
        "port": 5432,
        "database": "test_db",
        "user": "test_user",
        "password": "test_password",
    }
    timeout = 10
    query = "SELECT * FROM table_name;"
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    mock_pd_read_sql_query.return_value = df

    results_df = query_postgres_db(query, db_name, db_creds, timeout)
    assert mock_pd_read_sql_query.call_count == 1
    assert_frame_equal(results_df, df)


# test date/datetime type conversion to dateframe
@mock.patch("snowflake.connector.connect")
def test_query_snowflake_db(mock_connect):
    db_name = "test_db"
    db_creds = {
        "user": "test_user",
        "password": "test_password",
        "account": "test_account",
        "warehouse": "test_warehouse",
    }
    timeout = 10
    query = "SELECT * FROM table_name;"

    expected_df = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [4, 5, 6],
            "Date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]).date,
            "DateTime": pd.to_datetime(
                ["2023-01-01 10:00", "2023-01-02 11:00", "2023-01-03 12:00"]
            ),
        }
    )

    mock_cursor = mock.Mock()
    mock_connection = mock.Mock()

    mock_cursor.execute.return_value = None
    import datetime

    mock_cursor.fetchall.return_value = [
        [1, 4, datetime.date(2023, 1, 1), datetime.datetime(2023, 1, 1, 10, 0)],
        [2, 5, datetime.date(2023, 1, 2), datetime.datetime(2023, 1, 2, 11, 0)],
        [3, 6, datetime.date(2023, 1, 3), datetime.datetime(2023, 1, 3, 12, 0)],
    ]
    mock_cursor.description = [("A",), ("B",), ("Date",), ("DateTime",)]

    mock_connection.cursor.return_value = mock_cursor

    mock_connect.return_value = mock_connection

    results_df = query_snowflake_db(query, db_name, db_creds, timeout)

    assert mock_connect.call_count == 1
    mock_cursor.execute.assert_has_calls(
        [
            mock.call(f"USE WAREHOUSE {db_creds['warehouse']}"),
            mock.call(f"USE DATABASE {db_name}"),
            mock.call(query),
        ]
    )
    assert_frame_equal(results_df, expected_df)


def test_compare_df(test_dataframes):
    # Assigning the test_dataframes fixture to individual variables
    (
        df0,
        df0_same,
        df1,
        df2,
        df1_same,
        df1_value_diff,
        df1_columns_reordered,
        df1_columns_diffcase,
        df1_columns_renamed,
        df1_rows_reordered,
        df1_rows_reordered_columns_renamed,
        df1_rows_reordered_more_cols,
        df2_rows_reordered,
        df2_rows_reordered_sortb,
        df2_rows_reordered_sortb_renamed,
        df2_rows_reordered_sortb_more_cols,
    ) = test_dataframes

    question = "Here is a random question"
    question_sort = "Here is a random question that has a sort by instruction."
    sql_sort_b = "SELECT * FROM table_name ORDER BY B DESC"

    # Test case 1: Empty DataFrames, expect True
    assert compare_df(df0, df0_same, query, question) == True

    # Test case 2: Identical DataFrames, expect True
    assert compare_df(df1, df1_same, query, question) == True

    # Test case 3: Value Difference in a Column, expect False
    assert compare_df(df1, df1_value_diff, query, question) == False

    # Test case 4: Reordered Columns, expect True
    assert compare_df(df1, df1_columns_reordered, query, question) == True

    # Test case 5: Different Case in Column Names, expect True
    assert compare_df(df1, df1_columns_diffcase, query, question) == True

    # Test case 6: Renamed Columns, expect True
    assert compare_df(df1, df1_columns_renamed, query, question) == True

    # Test case 7: Reordered Rows, expect True
    assert compare_df(df1, df1_rows_reordered, query, question) == True

    # Test case 8: Reordered Rows with specific ordering, expect False
    assert compare_df(df1, df1_rows_reordered, "order_by", question_sort) == False

    # Test case 9: Reordered Rows with specific ordering and renamed columns, expect True
    assert (
        compare_df(df1, df1_rows_reordered_columns_renamed, query, question)
    ) == True

    # Test case 10: Reordered Rows with specific ordering and renamed and additional columns, expect False
    assert (compare_df(df1, df1_rows_reordered_more_cols, query, question)) == False

    # Test case 11: Reordered rows, expect True
    assert compare_df(df2, df2_rows_reordered, query, question) == True

    # Test case 12: Reordered rows with specific ordering, expect False
    assert compare_df(df2, df2_rows_reordered, "order_by", question_sort) == False

    # Test case 13: Reordered rows with specific asc ordering in col B, expect True
    assert (
        compare_df(
            df2,
            df2_rows_reordered_sortb,
            "order_by",
            question_sort,
            sql_sort_b,
            sql_sort_b,
        )
        == True
    )

    # Test case 14: Reordered rows with specific asc ordering in col B and renamed columns, expect True
    assert (
        compare_df(
            df2,
            df2_rows_reordered_sortb_renamed,
            "order_by",
            question_sort,
            sql_sort_b,
            sql_sort_b,
        )
        == True
    )

    # Test case 15: Reordered rows with specific asc ordering in col B and renamed and additional columns, expect False
    assert (
        compare_df(df2, df2_rows_reordered_sortb_more_cols, "order_by", question_sort)
        == False
    )


def test_subset_df(test_dataframes):
    # Assigning the test_dataframes fixture to individual variables
    (
        df0,
        df0_same,
        df1,
        df2,
        df1_same,
        df1_value_diff,
        df1_columns_reordered,
        df1_columns_diffcase,
        df1_columns_renamed,
        df1_rows_reordered,
        df1_rows_reordered_columns_renamed,
        df1_rows_reordered_more_cols,
        df2_rows_reordered,
        df2_rows_reordered_sortb,
        df2_rows_reordered_sortb_renamed,
        df2_rows_reordered_sortb_more_cols,
    ) = test_dataframes

    question = "Here is a random question"
    question_sort = "Here is a random question that has a sort by instruction."
    sql_sort_b = "SELECT * FROM table_name ORDER BY B ASC"

    # Test case 1: Empty DataFrames, expect False
    assert subset_df(df0, df0_same, query, question) == False

    # Test case 2: Identical DataFrames
    assert subset_df(df1, df1_same, query, question) == True

    # Test case 3: Value Difference in a Column
    assert subset_df(df1, df1_value_diff, query, question) == False

    # Test case 4: Reordered Columns
    assert subset_df(df1, df1_columns_reordered, query, question) == True

    # Test case 5: Different Case in Column Names
    assert subset_df(df1, df1_columns_diffcase, query, question) == True

    # Test case 6: Renamed Columns
    assert subset_df(df1, df1_columns_renamed, query, question) == True

    # Test case 7: Reordered Rows
    assert subset_df(df1, df1_rows_reordered, query, question) == True

    # Test case 8: Reordered Rows with specific ordering, expect False
    assert subset_df(df1, df1_rows_reordered, "order_by", question_sort) == False

    # Test case 9: Reordered Rows with specific ordering and renamed columns, expect True
    assert (subset_df(df1, df1_rows_reordered_columns_renamed, query, question)) == True

    # Test case 10: Reordered Rows with specific ordering and renamed and additional columns, expect True
    assert (subset_df(df1, df1_rows_reordered_more_cols, query, question)) == True

    # Test case 11: Reordered rows, expect True
    assert subset_df(df2, df2_rows_reordered, query, question) == True

    # Test case 12: Reordered rows with specific ordering, expect False
    assert subset_df(df2, df2_rows_reordered, "order_by", question_sort) == False

    # Test case 13: Reordered rows with specific asc ordering in col B, expect True
    result = subset_df(
        df2, df2_rows_reordered_sortb, "order_by", question_sort, sql_sort_b, sql_sort_b
    )
    assert result == True

    # Test case 14: Reordered rows with specific asc ordering in col B and renamed columns, expect True
    assert (
        subset_df(
            df2,
            df2_rows_reordered_sortb_renamed,
            "order_by",
            question_sort,
            sql_sort_b,
            sql_sort_b,
        )
        == True
    )

    # Test case 15: Reordered rows with specific asc ordering in col B and renamed and additional columns, expect True
    assert (
        subset_df(
            df2,
            df2_rows_reordered_sortb_more_cols,
            "order_by",
            question_sort,
            sql_sort_b,
            sql_sort_b,
        )
        == True
    )


@mock.patch("eval.eval.query_postgres_db")
def test_compare_query_results(mock_query_postgres_db):
    # Set up mock behavior
    def mock_query_postgres_db_fn(
        query, db_name, db_creds, timeout, decimal_points
    ) -> pd.DataFrame:
        if query == "SELECT id FROM users WHERE age < 18":
            return pd.DataFrame({"id": [1, 2, 3]})
        elif query == "SELECT name FROM users WHERE age < 18":
            return pd.DataFrame({"name": ["alice", "bob", "carol"]})
        elif query == "SELECT id, name FROM users WHERE age < 18":
            return pd.DataFrame({"id": [1, 2, 3], "name": ["alice", "bob", "carol"]})
        elif query == "SELECT id, age FROM users WHERE age < 18":
            return pd.DataFrame({"id": [1, 2, 3], "age": [16, 12, 17]})
        elif query == "SELECT id, age FROM users WHERE age < 18 ORDER BY age":
            return pd.DataFrame({"id": [2, 1, 3], "age": [12, 16, 17]})
        else:
            raise ValueError(f"Unexpected query: {query}")

    mock_query_postgres_db.side_effect = mock_query_postgres_db_fn

    query_gold = "SELECT {id,name} FROM users WHERE age < 18;"
    db_name = "test_db"
    db_type = "postgres"
    db_creds = {
        "host": "localhost",
        "port": 5432,
        "database": "test_db",
        "user": "test_user",
        "password": "test_password",
    }
    timeout = 10
    question = "Test question"
    query_category = "Test category"

    test_queries_expected = [
        (
            "SELECT id FROM users WHERE age < 18",
            True,
            True,
            2,
        ),  # uses id column (exact match on 1st column)
        (
            "SELECT name FROM users WHERE age < 18",
            True,
            True,
            3,
        ),  # uses name column (exact match on 2nd column)
        (
            "SELECT id, name FROM users WHERE age < 18",
            True,
            True,
            4,
        ),  # uses both columns (exact match on 3rd subset, both columns)
        (
            "SELECT id, age FROM users WHERE age < 18",
            False,
            True,
            4,
        ),  # returns additional columns (subset correct)
        (
            "SELECT id, age FROM users WHERE age < 18 ORDER BY age",
            False,
            True,
            4,
        ),  # returns additional columns in different row order (subset correct)
    ]

    for (
        query_gen,
        expected_same,
        expected_subset,
        expected_call_count,
    ) in test_queries_expected:
        print(f"evaluating query: {query_gen}")
        result_same, result_subset = compare_query_results(
            query_gold,
            query_gen,
            db_name,
            db_type,
            db_creds,
            question,
            query_category,
            timeout=timeout,
        )
        assert mock_query_postgres_db.call_count == expected_call_count
        assert result_same == expected_same
        assert result_subset == expected_subset
        # reset call count
        mock_query_postgres_db.reset_mock()
        mock_query_postgres_db.side_effect = mock_query_postgres_db_fn
