import pandas as pd

from eval.eval import get_all_minimal_queries, query_postgres_db


def test_questions_non_null():
    # read in questions_gen.csv
    df = pd.read_csv("data/questions_gen.csv")
    # for each row, run the query with eval.query_postgres_db
    # check that the result is not null
    for i, row in df.iterrows():
        queries_gold = get_all_minimal_queries(row["query"])
        for query_gold in queries_gold:
            df_result = query_postgres_db(query_gold, row["db_name"])
            if len(df_result) == 0:
                print(i, query_gold)
                print(df_result)
