import unittest
import json
import os
from query_generators.openai import OpenAIChatQueryGenerator


class TestGetMetadataSql(unittest.TestCase):
    def setUp(self):
        self.test_json_path = "/tmp/test.json"

        # Create a temporary JSON file for testing
        with open(self.test_json_path, "w") as f:
            json.dump(
                {
                    "table_metadata": {
                        "table1": [
                            {
                                "column_name": "column1",
                                "data_type": "int",
                                "column_description": "Description 1",
                            },
                            {
                                "column_name": "column2",
                                "data_type": "varchar",
                                "column_description": "Description 2",
                            },
                        ],
                        "table2": [
                            {
                                "column_name": "column3",
                                "data_type": "float",
                                "column_description": "Description 3",
                            }
                        ],
                    }
                },
                f,
            )

    def tearDown(self):
        # Delete the temporary JSON file after the test
        os.remove(self.test_json_path)

    def test_get_glossary_metadatasql(self):
        result = OpenAIChatQueryGenerator.get_metadata_sql(self.test_json_path)
        expected = """CREATE TABLE table1 (
  column1 int, --Description 1
  column2 varchar, --Description 2
)
-----------
CREATE TABLE table2 (
  column3 float, --Description 3
)
-----------
"""
        self.assertEqual(result, expected)
