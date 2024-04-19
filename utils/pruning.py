import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import spacy
import torch
import torch.nn.functional as F

if os.getenv("TOKENIZERS_PARALLELISM") is None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
nlp = spacy.load("en_core_web_sm")


def knn(
    query: str,
    all_emb: torch.tensor,
    k: int,
    threshold: float,
) -> Tuple[torch.tensor, torch.tensor]:
    """
    Get top most similar columns' embeddings to query using cosine similarity.
    """
    query_emb = encoder.encode(query, convert_to_tensor=True, device="cuda").unsqueeze(0)
    similarity_scores = F.cosine_similarity(query_emb, all_emb)
    top_results = torch.nonzero(similarity_scores > threshold).squeeze()
    # if top_results is empty, return empty tensors
    if top_results.numel() == 0:
        return torch.tensor([]), torch.tensor([])
    # if only 1 result is returned, we need to convert it to a tensor
    elif top_results.numel() == 1:
        return torch.tensor([similarity_scores[top_results]]), torch.tensor(
            [top_results]
        )
    else:
        top_k_scores, top_k_indices = torch.topk(
            similarity_scores[top_results], k=min(k, top_results.numel())
        )
        return top_k_scores, top_results[top_k_indices]


def get_entity_types(sentence, verbose: bool = False):
    """
    Get entity types from sentence using spaCy.
    """
    doc = nlp(sentence)
    named_entities = set()
    for ent in doc.ents:
        if verbose:
            print(f"ent {ent}, {ent.label_}")
        named_entities.add(ent.label_)

    return named_entities


def format_topk_sql(
    topk_table_columns: Dict[str, List[Tuple[str, str, str]]],
    shuffle: bool,
) -> str:
    if len(topk_table_columns) == 0:
        return ""
    md_str = "```\n"
    # shuffle the keys in topk_table_columns
    table_names = list(topk_table_columns.keys())
    if shuffle:
        np.random.seed(0)
        np.random.shuffle(table_names)
    for table_name in table_names:
        columns_str = ""
        columns = topk_table_columns[table_name]
        if shuffle:
            np.random.seed(0)
            np.random.shuffle(columns)
        for column_tuple in columns:
            if len(column_tuple) > 2:
                columns_str += (
                    f"\n  {column_tuple[0]} {column_tuple[1]}, --{column_tuple[2]}"
                )
            else:
                columns_str += f"\n  {column_tuple[0]} {column_tuple[1]}, "
        md_str += f"CREATE TABLE {table_name} ({columns_str}\n);\n"
    md_str += "```\n"
    return md_str


def get_md_emb(
    question: str,
    column_emb: torch.tensor,
    column_info_csv: List[str],
    column_ner: Dict[str, List[str]],
    column_join: Dict[str, dict],
    k: int,
    shuffle: bool,
    threshold: float = 0.2,
) -> str:
    """
    Given question, generated metadata csv string with top k columns and tables
    that are most similar to the question. `column_emb`, `column_info_csv`, `column_ner`,
    `column_join` are all specific to the db_name. `column_info_csv` is a list of csv strings
    with 1 row per column info, where each row is in the format:
    `table_name.column_name,column_type,column_description`.
    Steps are:
    1. Get top k columns from question to `column_emb` using `knn` and add the corresponding column info to topk_table_columns.
    2. Get entity types from question. If entity type is in `column_ner`, add the corresponding list of column info to topk_table_columns.
    3. Generate the metadata string using the column info so far, shuffling the order of the tables and the order of columns within the tables if `shuffle` is True.
    4. Get joinable columns between tables in topk_table_columns and add to final metadata string.
    """
    # 1) get top k columns
    top_k_scores, top_k_indices = knn(question, column_emb, k, threshold)
    topk_table_columns = {}
    table_column_names = set()
    for score, index in zip(top_k_scores, top_k_indices):
        table_name, column_info = column_info_csv[index].split(".", 1)
        column_tuple = tuple(column_info.split(",", 2))
        if table_name not in topk_table_columns:
            topk_table_columns[table_name] = []
        topk_table_columns[table_name].append(column_tuple)
        table_column_names.add(f"{table_name}.{column_tuple[0]}")

    # 2) get entity types from question + add corresponding columns
    entity_types = get_entity_types(question)
    for entity_type in entity_types:
        if entity_type in column_ner:
            for column_info in column_ner[entity_type]:
                table_column_name, column_type, column_description = column_info.split(
                    ",", 2
                )
                table_name, column_name = table_column_name.split(".", 1)
                if table_name not in topk_table_columns:
                    topk_table_columns[table_name] = []
                column_tuple = (column_name, column_type, column_description)
                if column_tuple not in topk_table_columns[table_name]:
                    topk_table_columns[table_name].append(column_tuple)
                table_column_names.add(table_column_name)
    topk_tables = sorted(list(topk_table_columns.keys()))

    # 3) get table pairs that can be joined
    # create dict of table_column_name -> column_tuple for lookups
    column_name_to_tuple = {}
    ncols = len(column_info_csv)
    for i in range(ncols):
        table_column_name, column_type, column_description = column_info_csv[i].split(
            ",", 2
        )
        table_name, column_name = table_column_name.split(".", 1)
        column_tuple = (column_name, column_type, column_description)
        column_name_to_tuple[table_column_name] = column_tuple
    # go through list of top k tables and see if pairs can be joined
    join_list = []
    for i in range(len(topk_tables)):
        for j in range(i + 1, len(topk_tables)):
            table1, table2 = topk_tables[i], topk_tables[j]
            assert table1 <= table2
            if (table1, table2) in column_join:
                for table_col_1, table_col_2 in column_join[(table1, table2)]:
                    # add to topk_table_columns
                    if table_col_1 not in table_column_names:
                        column_tuple = column_name_to_tuple[table_col_1]
                        topk_table_columns[table1].append(column_tuple)
                        table_column_names.add(table_col_1)
                    if table_col_2 not in table_column_names:
                        column_tuple = column_name_to_tuple[table_col_2]
                        topk_table_columns[table2].append(column_tuple)
                        table_column_names.add(table_col_2)
                    # add to join_list
                    join_str = f"{table_col_1} can be joined with {table_col_2}"
                    if join_str not in join_list:
                        join_list.append(join_str)

    # 4) format metadata string
    md_str = format_topk_sql(topk_table_columns, shuffle)

    if len(join_list) > 0:
        md_str += "\nHere is a list of joinable columns:\n"
        md_str += "\n".join(join_list)
        md_str += "\n"
    return md_str


def prune_metadata_str(
    question, db_name, public_data: bool, columns_to_keep: int, shuffle: bool
):
    # current file dir
    root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    if public_data:
        import defog_data.supplementary as sup

        emb_path = os.path.join(root_dir, "data", "public_embeddings.pkl")
    else:
        raise Exception("Include path to private embeddings here")
    # only read first 2 elements of tuple returned, since private method might return more
    emb_tuple = sup.load_embeddings(emb_path)
    emb = emb_tuple[0]
    csv_descriptions = emb_tuple[1]
    try:
        table_metadata_csv = get_md_emb(
            question,
            emb[db_name],
            csv_descriptions[db_name],
            sup.columns_ner[db_name],
            sup.columns_join[db_name],
            columns_to_keep,
            shuffle,
        )
    except KeyError:
        if public_data:
            raise ValueError(f"DB name `{db_name}` not found in public data")
        else:
            raise ValueError(f"DB name `{db_name}` not found in private data")
    return table_metadata_csv


def to_prompt_schema(
    md: Dict[str, List[Dict[str, str]]], seed: Optional[int] = None
) -> str:
    """
    Return a DDL statement for creating tables from a metadata dictionary
    `md` has the following structure:
        {'table1': [
            {'column_name': 'col1', 'data_type': 'int', 'column_description': 'primary key'},
            {'column_name': 'col2', 'data_type': 'text', 'column_description': 'not null'},
            {'column_name': 'col3', 'data_type': 'text', 'column_description': ''},
        ],
        'table2': [
        ...
        ]},
    This is just for converting the dictionary structure of one's metadata into a string
    for pasting into prompts, and not meant to be used to initialize a database.
    seed is used to shuffle the order of the tables when not None
    """
    md_create = ""
    table_names = list(md.keys())
    if seed:
        np.random.seed(seed)
        np.random.shuffle(table_names)
    for table in table_names:
        md_create += f"CREATE TABLE {table} (\n"
        columns = md[table]
        if seed:
            np.random.seed(seed)
            np.random.shuffle(columns)
        for i, column in enumerate(columns):
            col_name = column["column_name"]
            # if column name has spaces, wrap it in double quotes
            if " " in col_name:
                col_name = f'"{col_name}"'
            dtype = column["data_type"]
            col_desc = column.get("column_description", "").replace("\n", " ")
            if col_desc:
                col_desc = f" --{col_desc}"
            if i < len(columns) - 1:
                md_create += f"  {col_name} {dtype},{col_desc}\n"
            else:
                # avoid the trailing comma for the last line
                md_create += f"  {col_name} {dtype}{col_desc}\n"
        md_create += ");\n"
    return md_create
