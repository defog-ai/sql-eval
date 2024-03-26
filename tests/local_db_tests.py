import pandas as pd
from unittest import mock
from eval.eval import get_all_minimal_queries, query_postgres_db, query_postgres_temp_db
from pandas.testing import assert_frame_equal


def test_questions_non_null():
    # read in questions_gen_postgres.csv
    df = pd.read_csv("data/questions_gen_postgres.csv")
    # for each row, run the query with eval.query_postgres_db
    # check that the result is not null
    for i, row in df.iterrows():
        queries_gold = get_all_minimal_queries(row["query"])
        for query_gold in queries_gold:
            df_result = query_postgres_db(query_gold, row["db_name"])
            if len(df_result) == 0:
                print(i, query_gold)
                print(df_result)


@mock.patch("pandas.read_sql_query")
def test_query_postgres_temp_db(mock_pd_read_sql_query):
    # note that we need to mock create_engine
    db_name = "db_temp"
    db_creds = {
        "host": "localhost",
        "port": 5432,
        "user": "postgres",
        "password": "postgres",
    }

    table_metadata_string = "CREATE TABLE table_name (A INT, B INT);"
    timeout = 10
    query = "SELECT * FROM table_name;"
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    mock_pd_read_sql_query.return_value = df

    results_df = query_postgres_temp_db(
        query, db_name, db_creds, table_metadata_string, timeout
    )
    assert mock_pd_read_sql_query.call_count == 1
    assert_frame_equal(results_df, df)
