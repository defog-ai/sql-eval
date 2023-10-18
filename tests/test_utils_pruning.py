import pytest
from utils.pruning import encoder, get_entity_types, get_md_emb


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


# test embedding results + ner + join columns for sql
def test_get_md_emb(test_metadata):
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

Additionally, the following are tables/column pairs that can be joined in this database:
```
airport.country_id can be joined with country.id
```"""
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
        threshold,
    )
    assert result == ""
