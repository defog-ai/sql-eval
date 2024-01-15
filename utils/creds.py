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
}

bq_project = os.environ.get("BQ_PROJECT")
