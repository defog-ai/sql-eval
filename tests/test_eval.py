from eval.eval import (
    compare_query_results,
    find_bracket_indices,
    get_all_minimal_queries,
    normalize_table,
    compare_df,
    query_postgres_db,
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
        "name": ["John", "Jane", "Alice"],
        "age": [25, 30, 35],
        "city": ["New York", "London", "Paris"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def test_dataframes():
    df0 = pd.DataFrame({"A": [], "B": []})
    df0_same = pd.DataFrame({"A": [], "B": []})
    df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
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
    return (
        df0,
        df0_same,
        df1,
        df1_same,
        df1_value_diff,
        df1_columns_reordered,
        df1_columns_diffcase,
        df1_columns_renamed,
        df1_rows_reordered,
        df1_rows_reordered_columns_renamed,
        df1_rows_reordered_more_cols,
    )


def test_normalize_table_no_order_by(unordered_dataframe):
    # Test normalization without an order by clause
    expected_df = pd.DataFrame(
        {
            "age": [25, 30, 35],
            "city": ["New York", "London", "Paris"],
            "name": ["John", "Jane", "Alice"],
        }
    )
    question = "What is the average age of the people in the table?"
    normalized_df = normalize_table(unordered_dataframe, query, question)
    assert_frame_equal(expected_df, normalized_df)


def test_normalize_table_with_order_by(unordered_dataframe):
    # Test normalization with an order by clause
    expected_df = pd.DataFrame(
        {
            "age": [25, 30, 35],
            "city": ["New York", "London", "Paris"],
            "name": ["John", "Jane", "Alice"],
        }
    )
    question_sort = "What is the average age of the people in the table? sort by name."
    normalized_df = normalize_table(unordered_dataframe, query_order_by, question_sort)

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


def test_compare_df(test_dataframes):
    # Assigning the test_dataframes fixture to individual variables
    (
        df0,
        df0_same,
        df1,
        df1_same,
        df1_value_diff,
        df1_columns_reordered,
        df1_columns_diffcase,
        df1_columns_renamed,
        df1_rows_reordered,
        df1_rows_reordered_columns_renamed,
        df1_rows_reordered_more_cols,
    ) = test_dataframes

    question = "What is the average age of the people in the table?"
    question_sort = "What is the average age of the people in the table? sort by name."

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
    assert compare_df(df1, df1_rows_reordered, query_order_by, question_sort) == False

    # Test case 9: Reordered Rows with specific ordering and renamed columns, expect True
    assert (
        compare_df(df1, df1_rows_reordered_columns_renamed, query, question)
    ) == True

    # Test case 10: Reordered Rows with specific ordering and renamed and additional columns, expect False
    assert (compare_df(df1, df1_rows_reordered_more_cols, query, question)) == False


def test_subset_df(test_dataframes):
    # Assigning the test_dataframes fixture to individual variables
    (
        df0,
        df0_same,
        df1,
        df1_same,
        df1_value_diff,
        df1_columns_reordered,
        df1_columns_diffcase,
        df1_columns_renamed,
        df1_rows_reordered,
        df1_rows_reordered_columns_renamed,
        df1_rows_reordered_more_cols,
    ) = test_dataframes

    question = "What is the average age of the people in the table?"
    question_sort = "What is the average age of the people in the table? sort by name."

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
    assert subset_df(df1, df1_rows_reordered, query_order_by, question_sort) == False

    # Test case 9: Reordered Rows with specific ordering and renamed columns, expect True
    assert (subset_df(df1, df1_rows_reordered_columns_renamed, query, question)) == True

    # Test case 10: Reordered Rows with specific ordering and renamed and additional columns, expect True
    assert (subset_df(df1, df1_rows_reordered_more_cols, query, question)) == True


@mock.patch("eval.eval.query_postgres_db")
def test_compare_query_results(mock_query_postgres_db):
    # Set up mock behavior
    def mock_query_postgres_db_fn(query, db_name, db_creds, timeout) -> pd.DataFrame:
        if query == "SELECT id FROM users WHERE age < 18;":
            return pd.DataFrame({"id": [1, 2, 3]})
        elif query == "SELECT name FROM users WHERE age < 18;":
            return pd.DataFrame({"name": ["alice", "bob", "carol"]})
        elif query == "SELECT id, name FROM users WHERE age < 18;":
            return pd.DataFrame({"id": [1, 2, 3], "name": ["alice", "bob", "carol"]})
        elif query == "SELECT id, age FROM users WHERE age < 18;":
            return pd.DataFrame({"id": [1, 2, 3], "age": [16, 12, 17]})
        elif query == "SELECT id, age FROM users WHERE age < 18 ORDER BY age;":
            return pd.DataFrame({"id": [2, 1, 3], "age": [12, 16, 17]})
        else:
            raise ValueError("Unexpected query")

    mock_query_postgres_db.side_effect = mock_query_postgres_db_fn

    query_gold = "SELECT {id,name} FROM users WHERE age < 18;"
    db_name = "test_db"
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
            "SELECT id FROM users WHERE age < 18;",
            True,
            True,
            2,
        ),  # uses id column (exact match on 1st column)
        (
            "SELECT name FROM users WHERE age < 18;",
            True,
            True,
            3,
        ),  # uses name column (exact match on 2nd column)
        (
            "SELECT id, name FROM users WHERE age < 18;",
            True,
            True,
            4,
        ),  # uses both columns (exact match on 3rd subset, both columns)
        (
            "SELECT id, age FROM users WHERE age < 18;",
            False,
            True,
            4,
        ),  # returns additional columns (subset correct)
        (
            "SELECT id, age FROM users WHERE age < 18 ORDER BY age;",
            False,
            True,
            4,
        ),  # returns additional columns in different order (subset correct)
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
