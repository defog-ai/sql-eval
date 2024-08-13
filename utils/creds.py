import os

db_creds_all = {
    "postgres": {
        "host": os.environ.get("DBHOST", "localhost"),
        "port": os.environ.get("DBPORT", 5432),
        "user": os.environ.get("DBUSER", "postgres"),
        "password": os.environ.get("DBPASSWORD", "postgres"),
    },
    "snowflake": {
        "user": os.environ.get("SFDBUSER"),
        "password": os.environ.get("SFDBPASSWORD"),
        "account": os.environ.get("SFDBACCOUNT"),
        "warehouse": os.environ.get("SFDBWAREHOUSE"),
    },
    "mysql": {
        "user": "root",
        "password": "password",
        "host": "localhost",
    },
    "bigquery": {
        "project": os.environ.get("BIGQUERY_PROJ"),
        "creds": os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
    },
    "sqlite": {
        "path_to_folder": os.environ.get("HOME") + f"/defog-data/sqlite_dbs/",  # Path to folder containing sqlite dbs
    },
    "tsql": {
        "server": os.getenv("TSQL_SERVER"),
        "user": "test_user",
        "password": "password",
        "driver": "{ODBC Driver 17 for SQL Server}",
    },
}

bq_project = os.environ.get("BQ_PROJECT")
