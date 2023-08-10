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
    docker exec -i postgres-sql-eval psql -U postgres -c "DROP DATABASE IF EXISTS ${db_name};"
    docker exec -i postgres-sql-eval psql -U postgres -c "CREATE DATABASE ${db_name};"
    echo "done dropping and recreating database ${db_name}"
    docker_db_path="export/${db_name}.sql"
    docker exec -i postgres-sql-eval psql -U postgres -d "${db_name}" -f "${docker_db_path}"
done
