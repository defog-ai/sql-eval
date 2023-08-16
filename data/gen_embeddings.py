# for each db_name, get all of the column names and descriptions for each table and embed them

import json
import pickle
from sentence_transformers import SentenceTransformer

# load model for embedding column descriptions
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

db_names = [
    "academic",
    "atis",
    "advising",
    "geography",
    "restaurants",
    "scholar",
    "yelp",
    "pipeline",
]
emb = {}
csv_descriptions = {}
for db_name in db_names:
    with open(f"auto_eval/data/metadata/{db_name}.json", "r") as f:
        metadata = json.load(f)["table_metadata"]
    column_descriptions = []
    column_descriptions_typed = []
    for table in metadata:
        for column in metadata[table]:
            col_str = (
                table
                + "."
                + column["column_name"]
                + ": "
                + column["column_description"]
            )
            col_str_typed = (
                table
                + "."
                + column["column_name"]
                + ","
                + column["data_type"]
                + ","
                + column["column_description"]
            )
            column_descriptions.append(col_str)
            column_descriptions_typed.append(col_str_typed)
    column_emb = encoder.encode(column_descriptions, convert_to_tensor=True)
    emb[db_name] = column_emb
    csv_descriptions[db_name] = column_descriptions_typed
    print(f"Finished embedding {db_name} {len(column_descriptions)} columns")

with open("data/embeddings.pkl", "wb") as f:
    pickle.dump((emb, csv_descriptions), f)

# entity types: list of (column, type, description) tuples
# note that these are spacy types https://spacy.io/usage/linguistic-features#named-entities
# we can add more types if we want, but PERSON, GPE, ORG should be
# sufficient for most use cases.
# also note that DATE and TIME are not included because they are usually
# retrievable from the top k embedding search due to the limited list of nouns
columns_ner = {
    "academic": {
        "PERSON": [
            "author.name,text,The name of the author",
        ],
        "ORG": [
            "conference.name,text,The name of the conference",
            "journal.name,text,The name of the journal",
            "organization.name,text,The name of the organization",
        ],
    },
    "advising": {
        "PERSON": [
            "instructor.name,text,The name of the instructor",
            "student.firstname,text,The first name of the student",
            "student.lastname,text,The last name of the student",
        ],
        "ORG": [
            "program.college,text,Name of the college offering the program",
            "program.name,text,Name of the program",
        ],
    },
    "atis": {
        "GPE": [
            "city.city_code,text,The city code",
            "city.city_name,text,The city name",
            "city.state_code,text,The state code",
            "city.country_name,text,The country name",
            "state.state_code,text,The state code",
            "state.state_name,text,The state name",
            "state.country_name,text,The country name",
            "airport.airport_location,text,The airport location",
            "airport.country_name,text,The country the airport is located in",
            "airport.state_code,text,The state the airport is located in",
            "flight_stop.stop_airport,text,The 3-letter airport code for the stop",
            "ground_service.city_code,text,The city code where ground service is available",
            "ground_service.airport_code,text,The airport code where ground service is available",
            "airport_service.city_code,text,The city code where airport service is available",
            "airport_service.airport_code,text,The airport code where airport service is available",
        ],
        "ORG": [
            "fare.fare_airline,text,The airline's name",
            "fare.from_airport,text,The 3-letter airport code for the departure location",
            "fare.to_airport,text,The 3-letter airport code for the arrival location",
            "flight.airline_code,text,Code assigned to airline",
            "flight.from_airport,text,The 3-letter airport code for the departure location",
            "flight.to_airport,text,The 3-letter airport code for the arrival location",
            "flight.airline_flight,text,Code assigned to the flight",
            "airline.airline_code,text,Code assigned to airline",
            "airline.airline_name,text,The airline's name",
            "airport.airport_name,text,The name of the airport",
            "airport.airport_code,text,The 3-letter airport code",
            "dual_carrier.main_airline,text,The name of the main airline operating the flight",
        ],
    },
    "pipeline": {
        "GPE": [
            "ps_studios.country,text,The country where the studio is located",
            "ps_studios.region,text,The region where the studio is located",
            "ps_studios.state_province,text,The state/province where the studio is located",
            "ps_studios.city,text,The city where the studio is located",
        ],
        "ORG": [
            "ps_studios.name,text,The name of the studio",
        ],
        "PER": [
            "clients.first_name,text,The first name of the client",
            "clients.last_name,text,The last name of the client",
        ],
    },
    "yelp": {
        "GPE": [
            "business.city,text,The name of the city where the business is located",
            "business.state,text,The US state where the business is located. This is represented by two-letter state abbreviations.",
            "business.full_address,text,The full address of the business",
        ],
        "ORG": [
            "business.name,text,The name of the business",
            "neighbourhood.neighbourhood_name,text,Name of the neighbourhood where the business is located",
        ],
        "PER": [
            "users.name,text,The name of the user",
        ],
    },
    "restaurants": {
        "GPE": [
            "location.city_name,text,The city where the restaurant is located",
            "location.street_name,text,The street where the restaurant is located",
            "geographic.city_name,text,The city where the restaurant is located",
            "geographic.county,text,The county where the restaurant is located",
            "geographic.region,text,The region where the restaurant is located",
            "restaurant.city_name,text,The city where the restaurant is located",
        ],
        "ORG": [
            "restaurant.name,text,The name of the restaurant",
            "restaurant.id,bigint,The ID of the restaurant",
            "restaurant.name,text,The name of the restaurant",
        ],
        "PER": [],
    },
    "geography": {
        "GPE": [
            "city.city_name,text,The name of the city",
            "city.country_name,text,The name of the country",
            "city.state_name,text,The name of the state",
            "lake.country_name,text,The name of the country where the lake is located",
            "lake.state_name,text,The name of the state where the lake is located (if applicable)",
            "river.country_name,text,The name of the country where the river flows through",
            "river.traverse, text, The cities or landmarks the river passes through. Comma delimited, eg `new york,albany,boston`",
            "state.state_name,text,The name of the state",
            "state.country_name,text,The name of the country the state belongs to",
            "state.capital,text,The name of the capital city of the state",
            "highlow.state_name,text,The name of the state",
            "mountain.country_name,text,The name of the country where the mountain is located",
            "mountain.state_name,text,The name of the state where the mountain is located (if applicable)",
            "border_info.state_name,text,The name of the state that shares a border with another state or country",
            "border_info.border,text,The name of the state or country that shares a border with the state specified in the state_name column",
        ],
        "LOC": [
            "lake.lake_name,text,The name of the lake",
            "river.river_name,text,The name of the river",
            "mountain.mountain_name,text,The name of the mountain",
        ],
        "ORG": [],
        "PER": [],
    },
    "scholar": {
        "GPE": [],
        "EVENT": [
            "venue.venuename,text,Name of the venue",
        ],
        "ORG": [],
        "PER": [
            "author.authorname,text,Name of the author",
        ],
        "WORK_OF_ART": [
            "paper.title,text,The title of the paper, enclosed in double quotes if it contains commas.",
            "dataset.datasetname,text,Name of the dataset",
            "journal.journalname,text,Name or title of the journal",
        ],
    },
}

# (pair of tables): list of (column1, column2) tuples that can be joined
# pairs should be lexically ordered, ie (table1 < table2) and (column1 < column2)
columns_join = {
    "academic": {
        ("author", "domain_author"): [("author.aid", "domain_author.aid")],
        ("author", "organization"): [("author.oid", "organization.oid")],
        ("author", "writes"): [("author.aid", "writes.aid")],
        ("cite", "publication"): [
            ("cite.cited", "publication.pid"),
            ("cite.citing", "publication.pid"),
        ],
        ("conference", "domain_conference"): [
            ("conference.cid", "domain_conference.cid")
        ],
        ("conference", "publication"): [("conference.cid", "publication.cid")],
        ("domain", "domain_author"): [("domain.did", "domain_author.did")],
        ("domain", "domain_conference"): [("domain.did", "domain_conference.did")],
        ("domain", "domain_journal"): [("domain.did", "domain_journal.did")],
        ("domain", "domain_keyword"): [("domain.did", "domain_keyword.did")],
        ("domain_journal", "journal"): [("domain_journal.jid", "journal.jid")],
        ("domain_keyword", "keyword"): [("domain_keyword.kid", "keyword.kid")],
        ("domain_publication", "publication"): [
            ("domain_publication.pid", "publication.pid")
        ],
        ("journal", "publication"): [("journal.jid", "publication.jid")],
        ("keyword", "publication_keyword"): [
            ("keyword.kid", "publication_keyword.kid")
        ],
        ("publication", "publication_keyword"): [
            ("publication.pid", "publication_keyword.pid")
        ],
        ("publication", "writes"): [("publication.pid", "writes.pid")],
    },
    "advising": {
        ("area", "course"): [("area.course_id", "course.course_id")],
        ("comment_instructor", "instructor"): [
            ("comment_instructor.instructor_id", "instructor.instructor_id")
        ],
        ("comment_instructor", "student"): [
            ("comment_instructor.student_id", "student.student_id")
        ],
        ("course", "course_offering"): [
            ("course.course_id", "course_offering.course_id")
        ],
        ("course", "course_prerequisite"): [
            ("course.course_id", "course_prerequisite.course_id"),
            ("course.course_id", "course_prerequisite.pre_course_id"),
        ],
        ("course", "course_tags_count"): [
            ("course.course_id", "course_tags_count.course_id")
        ],
        ("course", "program_course"): [
            ("course.course_id", "program_course.course_id")
        ],
        ("course", "student_record"): [
            ("course.course_id", "student_record.course_id")
        ],
        ("course_offering", "gsi"): [
            ("course_offering.offering_id", "gsi.course_offering_id")
        ],
        ("course_offering", "offering_instructor"): [
            ("course_offering.offering_id", "offering_instructor.offering_id")
        ],
        ("course_offering", "student_record"): [
            ("course_offering.offering_id", "student_record.offering_id"),
            ("course_offering.course_id", "student_record.course_id"),
        ],
        ("instructor", "offering_instructor"): [
            ("instructor.instructor_id", "offering_instructor.instructor_id")
        ],
        ("program", "program_course"): [
            ("program.program_id", "program_course.program_id")
        ],
        ("program", "program_requirement"): [
            ("program.program_id", "program_requirement.program_id")
        ],
        ("program", "student"): [("program.program_id", "student.program_id")],
        ("student", "student_record"): [
            ("student.student_id", "student_record.student_id")
        ],
    },
    "atis": {
        ("airline", "flight"): [("airline.airline_code", "flight.airline_code")],
        ("airline", "flight_stop"): [
            ("airline.airline_code", "flight_stop.departure_airline"),
            ("airline.airline_code", "flight_stop.arrival_airline"),
        ],
        ("airport", "fare"): [
            ("airport.airport_code", "fare.from_airport"),
            ("airport.airport_code", "fare.to_airport"),
        ],
        ("airport", "flight_stop"): [
            ("airport.airport_code", "flight_stop.stop_airport")
        ],
        ("airport_service", "ground_service"): [
            ("airport_service.city_code", "ground_service.city_code"),
            ("airport_service.airport_code", "ground_service.airport_code"),
        ],
        ("airport", "city"): [
            ("airport.state_code", "city.state_code"),
            ("airport.country_name", "city.state_code"),
            ("airport.time_zone_code", "city.state_code"),
        ],
    },
    "pipeline": {
        ("ps_studios", "sales"): [("ps_studios.site_id", "sales.site_id")],
        ("ps_studios", "visits"): [("ps_studios.site_id", "visits.site_id")],
        ("clients", "ps_studios"): [("clients.site_id", "ps_studios.site_id")],
        ("sales", "visits"): [("sales.client_id", "visits.client_id"), ("sales.site_id", "visits.site_id")],
        ("clients", "sales"): [("clients.client_id", "sales.client_id"), ("clients.client_id", "sales.recipient_client_id"), ("clients.site_id", "sales.site_id")],
        ("clients", "visits"): [("clients.client_id", "visits.client_id"), ("clients.site_id", "visits.site_id")],
    },
    "yelp": {
        ("business", "tip"): [("business.business_id", "tip.business_id")],
        ("business", "review"): [("business.business_id", "review.business_id")],
        ("business", "checkin"): [("business.business_id", "checkin.business_id")],
        ("business", "neighbourhood"): [
            ("business.business_id", "neighbourhood.business_id")
        ],
        ("business", "category"): [("business.business_id", "category.business_id")],
        ("tip", "users"): [("tip.user_id", "users.user_id")],
        ("review", "users"): [("review.user_id", "users.user_id")],
    },
    "restaurants": {
        ("geographic", "location"): [
            ("geographic.city_name", "location.city_name"),
        ],
        ("geographic", "restaurant"): [
            ("geographic.city_name", "restaurant.city_name"),
        ],
        ("location", "restaurant"): [
            ("location.restaurant_id", "restaurant.id"),
        ],
    },
    "geography": {
        ("border_info", "city"): [
            ("border_info.state_name", "city.state_name"),
            ("border_info.border", "city.state_name"),
        ],
        ("border_info", "lake"): [
            ("border_info.state_name", "lake.state_name"),
            ("border_info.border", "lake.state_name"),
        ],
        ("border_info", "state"): [
            ("border_info.state_name", "state.state_name"),
            ("border_info.border", "state.state_name"),
        ],
        ("border_info", "highlow"): [
            ("border_info.state_name", "highlow.state_name"),
            ("border_info.border", "highlow.state_name"),
        ],
        ("border_info", "mountain"): [
            ("border_info.state_name", "mountain.state_name"),
            ("border_info.border", "mountain.state_name"),
        ],
        ("city", "lake"): [
            ("city.country_name", "lake.country_name"),
            ("city.state_name", "lake.state_name"),
        ],
        ("city", "river"): [
            ("city.country_name", "river.country_name"),
        ],
        ("city", "state"): [
            ("city.country_name", "state.country_name"),
            ("city.state_name", "state.state_name"),
        ],
        ("city", "mountain"): [
            ("city.country_name", "mountain.country_name"),
            ("city.state_name", "mountain.state_name"),
        ],
        ("city", "highlow"): [
            ("city.state_name", "highlow.state_name"),
        ],
        ("highlow", "lake"): [
            ("highlow.state_name", "lake.state_name"),
        ],
        ("highlow", "state"): [
            ("highlow.state_name", "state.state_name"),
        ],
        ("highlow", "mountain"): [
            ("highlow.state_name", "mountain.state_name"),
        ],
        ("lake", "river"): [
            ("lake.country_name", "river.country_name"),
        ],
        ("lake", "state"): [
            ("lake.country_name", "state.country_name"),
            ("lake.state_name", "state.state_name"),
        ],
        ("lake", "mountain"): [
            ("lake.country_name", "mountain.country_name"),
            ("lake.state_name", "mountain.state_name"),
        ],
        ("river", "state"): [
            ("river.country_name", "state.country_name"),
        ],
        ("river", "mountain"): [
            ("river.country_name", "mountain.country_name"),
        ],
        ("state", "mountain"): [
            ("state.country_name", "mountain.country_name"),
            ("state.state_name", "mountain.state_nem"),
        ],
    },
    "scholar": {
        ("author", "writes"): [
            ("author.authorid", "writes.authorid"),
        ],
        ("cite", "paper"): [
            ("cite.citingpaperid", "paper.paperid"),
            ("cite.citedpaperid", "paper.paperid"),
        ],
        ("cite", "paperdataset"): [
            ("cite.citingpaperid", "paperdataset.paperid"),
            ("cite.citedpaperid", "paperdataset.paperid"),
        ],
        ("cite", "paperfield"): [
            ("cite.citingpaperid", "paperfield.paperid"),
            ("cite.citedpaperid", "paperfield.paperid"),
        ],
        ("cite", "paperkeyphrase"): [
            ("cite.citingpaperid", "paperkeyphrase.paperid"),
            ("cite.citedpaperid", "paperkeyphrase.paperid"),
        ],
        ("cite", "writes"): [
            ("cite.citingpaperid", "writes.paperid"),
            ("cite.citedpaperid", "writes.paperid"),
        ],
        ("dataset", "paperdataset"): [
            ("dataset.datasetid", "paperdataset.datasetid"),
        ],
        ("field", "paperfield"): [
            ("field.fieldid", "paperfield.fieldid"),
        ],
        ("journal", "paper"): [
            ("journal.journalid", "paper.journalid"),
        ],
        ("keyphrase", "paperkeyphrase"): [
            ("keyphrase.keyphraseid", "paperkeyphrase.keyphraseid"),
        ],
        ("paper", "paperdataset"): [
            ("paper.paperid", "paperdataset.paperid"),
        ],
        ("paper", "paperfield"): [
            ("paper.paperid", "paperfield.paperid"),
        ],
        ("paper", "paperkeyphrase"): [
            ("paper.paperid", "paperkeyphrase.paperid"),
        ],
        ("paper", "writes"): [
            ("paper.paperid", "writes.paperid"),
        ],
        ("paper", "venue"): [
            ("paper.venueid", "venue.venueid"),
        ],
        ("paperfield", "paperkeyphrase"): [
            ("paperfield.paperid", "paperkeyphrase.paperid"),
        ],
        ("paperfield", "writes"): [
            ("paperfield.paperid", "writes.paperid"),
        ],
        ("paperkeyphrase", "writes"): [
            ("paperkeyphrase.paperid", "writes.paperid"),
        ],
    },
}

fpath = "data/ner_metadata.pkl"
with open(fpath, "wb") as f:
    pickle.dump((columns_ner, columns_join), f)
    print(f"Saved NER columns/tables and JOIN column relationships to {fpath}")