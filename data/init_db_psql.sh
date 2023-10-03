set -e

# get arguments
# if $@ is empty, set it to academic advising atis geography restaurants scholar yelp
if [ -z "$@" ]; then
    set -- academic advising atis geography restaurants scholar yelp
fi
# $@ is all arguments passed to the script
echo "Databases to init: $@"

# for each file name passed by the user, check if it exists in data/export
for db_name in "$@"; do
    db_path="data/export/${db_name}.sql"
    if [ ! -f "$db_path" ]; then
        echo "Database ${db_name} does not exist in data/export"
        exit 1
    fi
done


# get each folder name in data/export
for db_name in "$@"; do
    echo "dropping and recreating database ${db_name}"
    # drop and recreate database
    PGPASSWORD=$DBPASSWORD psql -U $DBUSER -h $DBHOST -p $DBPORT -c "DROP DATABASE IF EXISTS ${db_name};"
    PGPASSWORD=$DBPASSWORD psql -U $DBUSER -h $DBHOST -p $DBPORT -c "CREATE DATABASE ${db_name};"
    echo "done dropping and recreating database ${db_name}"
    db_path="data/export/${db_name}.sql"
    echo "importing ${db_path} into database ${db_name}"
    PGPASSWORD=$DBPASSWORD psql -U $DBUSER -h $DBHOST -p $DBPORT -d "${db_name}" -f "${db_path}"
done

# get the default embeddings and ner metadata pkl files from GCS bucket, and save them here
curl -L https://storage.googleapis.com/defog-ai/embeddings.pkl -o data/embeddings.pkl
curl -L https://storage.googleapis.com/defog-ai/ner_metadata.pkl -o data/ner_metadata.pkl