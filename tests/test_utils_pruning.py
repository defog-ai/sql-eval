import pytest
from utils.pruning import (
    encoder,
    get_entity_types,
    format_topk_sql,
    get_md_emb,
    to_prompt_schema,
)


@pytest.fixture
def test_metadata():
    column_csv = [
        "country.name,text,country name",
        "country.capital,text,country capital",
        "country.id,integer,unique id for country, not iso code",
        "airport.country_id,integer,unique id for country where airport is located in",
        "airport.airport_name,text,name of airport",
        "flight.pilot_name,text,name of the pilot",
        "flight.airport_name,text,name of the airport",
        "flight.flight_code,text,flight code",
    ]
    column_emb = encoder.encode(column_csv, convert_to_tensor=True)
    column_ner = {
        "GPE": [
            "country.name,text,country name",
            "country.capital,text,country capital",
            "airport.country_name,text,name of the country where the airport is located in",
        ],
        "ORG": [
            "country.name,text,country name",
            "airport.airport_name,text,name of airport",
            "flight.airport_name,text,name of the airport",
        ],
        "PERSON": ["flight.pilot_name,text,name of the pilot"],
    }
    column_join = {("airport", "country"): [("airport.country_id", "country.id")]}
    return column_emb, column_csv, column_ner, column_join


@pytest.fixture
def test_metadata_diff_coldesc():
    column_csv = [
        "country.name,text,country name",
        "country.capital,text,country capital",
        "country.id,integer,unique id for country, not iso code",
        "airport.country_id,integer,unique id for country where airport is located in",
        "airport.country_name,text,name of the country where the airport is located in",  # added
        "airport.airport_name,text,name of airport",
        "flight.pilot_name,text,name of the pilot",
        "flight.airport_name,text,name of the airport",
        "flight.flight_code,text,flight code",
    ]
    column_emb = encoder.encode(column_csv, convert_to_tensor=True)
    column_ner = {
        "GPE": [
            "country.name,text,country name",
            "country.capital,text,country capital",
            "airport.country_name,text,name of the country where the airport is located in",
        ],
        "ORG": [
            "country.name,text,country name",
            "airport.country_name,text,name of the country where the airport is located in",  # added
            "airport.airport_name,text,name of airport",
            "flight.airport_name,text,name of the airport",
        ],
        "PERSON": ["flight.pilot_name,text,name of the pilot"],
    }
    column_join = {
        ("airport", "country"): [
            ("airport.country_id", "country.id"),
            ("airport.country_name", "country.name"),
        ]
    }
    return column_emb, column_csv, column_ner, column_join


# test embedding results + ner + join columns for sql
def test_get_md_emb_no_shuffle(test_metadata):
    column_emb, column_csv, column_ner, column_join = test_metadata
    question = "How many flights start from Los Angeles Airport (LAX)?"
    assert get_entity_types(question) == {"GPE", "ORG"}
    k = 3
    threshold = 0.0

    # Call the function and get the result
    result = get_md_emb(
        question,
        column_emb,
        column_csv,
        column_ner,
        column_join,
        k,
        False,
        threshold,
    )
    print(f"result\n{result}")
    expected = """```
CREATE TABLE flight (
  airport_name text, --name of the airport
  flight_code text, --flight code
);
CREATE TABLE airport (
  airport_name text, --name of airport
  country_name text, --name of the country where the airport is located in
  country_id integer, --unique id for country where airport is located in
);
CREATE TABLE country (
  name text, --country name
  capital text, --country capital
  id integer, --unique id for country, not iso code
);
```

Here is a list of joinable columns:
airport.country_id can be joined with country.id
"""
    assert result == expected


def test_get_md_emb_shuffle(test_metadata):
    column_emb, column_csv, column_ner, column_join = test_metadata
    question = "How many flights start from Los Angeles Airport (LAX)?"
    assert get_entity_types(question) == {"GPE", "ORG"}
    k = 3
    threshold = 0.0

    # Call the function and get the result
    result = get_md_emb(
        question,
        column_emb,
        column_csv,
        column_ner,
        column_join,
        k,
        True,
        threshold,
    )
    print(f"result\n{result}")
    expected = """```
CREATE TABLE country (
  id integer, --unique id for country, not iso code
  capital text, --country capital
  name text, --country name
);
CREATE TABLE airport (
  country_id integer, --unique id for country where airport is located in
  country_name text, --name of the country where the airport is located in
  airport_name text, --name of airport
);
CREATE TABLE flight (
  flight_code text, --flight code
  airport_name text, --name of the airport
);
```

Here is a list of joinable columns:
airport.country_id can be joined with country.id
"""
    assert result == expected


def test_get_md_emb_sql_emb_empty(test_metadata):
    column_emb, column_csv, column_ner, column_join = test_metadata
    question = "Who ate my homework?"
    k = 3
    threshold = 1.0  # arbitrarily high threshold to test empty results

    # Call the function and get the result
    result = get_md_emb(
        question,
        column_emb,
        column_csv,
        column_ner,
        column_join,
        k,
        False,
        threshold,
    )
    assert result == ""


def test_get_md_emb_coldesc(test_metadata_diff_coldesc):
    column_emb, column_csv, column_ner, column_join = test_metadata_diff_coldesc
    question = "How many flights start from Los Angeles Airport (LAX)?"
    assert get_entity_types(question) == {"GPE", "ORG"}
    k = 3
    threshold = 0.0

    # Call the function and get the result
    result = get_md_emb(
        question,
        column_emb,
        column_csv,
        column_ner,
        column_join,
        k,
        False,
        threshold,
    )
    print(f"result\n{result}")
    # count "name text" in the result
    assert (
        result.count(
            "country_name text, --name of the country where the airport is located in"
        )
        == 1
    )


def test_format_topk_sql_empty():
    assert format_topk_sql({}, False) == ""


def test_format_topk_sql_single_table():
    table_columns = {
        "table1": [("column1", "type1", "comment1"), ("column2", "type2", "comment2")]
    }
    expected_output = (
        "```\n"
        "CREATE TABLE table1 (\n"
        "  column1 type1, --comment1\n"
        "  column2 type2, --comment2\n"
        ");\n"
        "```\n"
    )
    assert format_topk_sql(table_columns, False) == expected_output


def test_format_topk_sql_multiple_tables():
    table_columns = {
        "table1": [("column1", "type1", "comment1"), ("column2", "type2", "comment2")],
        "table2": [("column3", "type3", "comment3"), ("column4", "type4", "comment4")],
    }
    expected_output = (
        "```\n"
        "CREATE TABLE table1 (\n"
        "  column1 type1, --comment1\n"
        "  column2 type2, --comment2\n"
        ");\n"
        "CREATE TABLE table2 (\n"
        "  column3 type3, --comment3\n"
        "  column4 type4, --comment4\n"
        ");\n"
        "```\n"
    )
    assert format_topk_sql(table_columns, False) == expected_output


def test_format_topk_sql_shuffle():
    table_columns = {
        "table1": [("column1", "type1", "comment1"), ("column2", "type2", "comment2")],
        "table2": [("column3", "type3", "comment3"), ("column4", "type4", "comment4")],
    }
    # Since the shuffle operation is deterministic (we set the seed to 0),
    # we can still predict the output.
    expected_output = (
        "```\n"
        "CREATE TABLE table2 (\n"
        "  column4 type4, --comment4\n"
        "  column3 type3, --comment3\n"
        ");\n"
        "CREATE TABLE table1 (\n"
        "  column2 type2, --comment2\n"
        "  column1 type1, --comment1\n"
        ");\n"
        "```\n"
    )
    assert format_topk_sql(table_columns, True) == expected_output


def test_to_prompt_schema_without_seed():
    metadata = {
        "table1": [
            {
                "column_name": "col1",
                "data_type": "int",
                "column_description": "primary key",
            },
            {
                "column_name": "col2",
                "data_type": "text",
                "column_description": "not null",
            },
            {"column_name": "col3", "data_type": "text", "column_description": ""},
        ],
        "table2": [
            {
                "column_name": "col1",
                "data_type": "int",
                "column_description": "primary key",
            },
            {
                "column_name": "col2",
                "data_type": "text",
                "column_description": "not null",
            },
        ],
    }
    result = to_prompt_schema(metadata)
    expected = """CREATE TABLE table1 (
  col1 int, --primary key
  col2 text, --not null
  col3 text
);
CREATE TABLE table2 (
  col1 int, --primary key
  col2 text --not null
);
"""
    assert result == expected


def test_to_prompt_schema_with_seed():
    metadata = {
        "table1": [
            {
                "column_name": "col1",
                "data_type": "int",
                "column_description": "primary key",
            },
            {
                "column_name": "col2",
                "data_type": "text",
                "column_description": "not null",
            },
            {"column_name": "col3", "data_type": "text", "column_description": ""},
        ],
        "table2": [
            {
                "column_name": "col1",
                "data_type": "int",
                "column_description": "primary key",
            },
            {
                "column_name": "col2",
                "data_type": "text",
                "column_description": "not null",
            },
        ],
    }
    result = to_prompt_schema(metadata, seed=1)
    expected = """CREATE TABLE table1 (
  col1 int, --primary key
  col3 text,
  col2 text --not null
);
CREATE TABLE table2 (
  col1 int, --primary key
  col2 text --not null
);
"""
    assert result == expected
