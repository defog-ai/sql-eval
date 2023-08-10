import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from eval.eval import normalize_table, compare_df, subset_df

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

    # Test case 5: Different Case in Column Names, assume already done so False
    assert compare_df(df1, df1_columns_diffcase, query, question) == False

    # Test case 6: Renamed Columns, expect False
    assert compare_df(df1, df1_columns_renamed, query, question) == False

    # Test case 7: Reordered Rows, expect True
    assert compare_df(df1, df1_rows_reordered, query, question) == True

    # Test case 8: Reordered Rows with specific ordering, expect False
    assert compare_df(df1, df1_rows_reordered, query_order_by, question_sort) == False

    # Test case 9: Reordered Rows with specific ordering and renamed columns, expect False
    assert (
        compare_df(df1, df1_rows_reordered_columns_renamed, query, question)
    ) == False

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

    # Test case 1: Empty DataFrames
    assert subset_df(df0, df0_same, query, question) == True

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
