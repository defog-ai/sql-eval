import re
from defog_data.metadata import dbs
import os
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
import spacy
import torch
import torch.nn.functional as F

if os.getenv("TOKENIZERS_PARALLELISM") is None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
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
    query_emb = encoder.encode(query, convert_to_tensor=True).unsqueeze(0)
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
) -> str:
    if len(topk_table_columns) == 0:
        return ""
    md_str = "```\n"
    for table_name in topk_table_columns:
        columns_str = ""
        for column_tuple in topk_table_columns[table_name]:
            if len(column_tuple) > 2:
                columns_str += (
                    f"\n  {column_tuple[0]} {column_tuple[1]}, --{column_tuple[2]}"
                )
            else:
                columns_str += f"\n  {column_tuple[0]} {column_tuple[1]}, "
        md_str += f"CREATE TABLE {table_name} ({columns_str}\n);\n"
    return md_str


def get_md_emb(
    question: str,
    column_emb: torch.tensor,
    column_info_csv: List[str],
    column_ner: Dict[str, List[str]],
    column_join: Dict[str, dict],
    k: int = 20,
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
    3. Generate the metadata string using the column info so far.
    4. Get joinable columns between tables in topk_table_columns and add to final metadata string.
    """
    if torch.cuda.is_available():
        column_emb = column_emb.to("cuda")
    else:
        column_emb = column_emb.to("cpu")
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
    md_str = format_topk_sql(topk_table_columns)

    if len(join_list) > 0:
        md_str += "```\n\nAdditionally, the following are tables/column pairs that can be joined in this database:\n```\n"
        md_str += "\n".join(join_list)
        md_str += "\n```"
    return md_str


def prune_metadata_str(question, db_name):
    # current file dir
    root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    # We assume that as long as the db_name is one of our 8 public datasets, it will use the
    # public dataset embeddings. Otherwise, it will use the private dataset embeddings.
    if db_name in dbs:
        import defog_data.supplementary as sup

        emb_path = os.path.join(root_dir, "data", "public_embeddings.pkl")
    else:
        import defog_data_private.supplementary as sup

        emb_path = os.path.join(root_dir, "data", "private_embeddings.pkl")
    # only read first 2 elements of tuple returned, since private method might return more
    emb_tuple = sup.load_embeddings(emb_path)
    emb = emb_tuple[0]
    csv_descriptions = emb_tuple[1]
    table_metadata_csv = get_md_emb(
        question,
        emb[db_name],
        csv_descriptions[db_name],
        sup.columns_ner[db_name],
        sup.columns_join[db_name],
    )
    return table_metadata_csv


def generate_prompt(prompt_file, question, db_name, instructions=None):
    """
    Given prompt template file path, question, and db_name, generate prompt.
    We will run the pruning algorithm to get the metadata string for the db_name.
    Instructions are optional, and if present, will have a section heading
    appended for clarity in the prompt.
    """
    with open(prompt_file, "r") as f:
        prompt = f.read()
    pruned_metadata_str = prune_metadata_str(question, db_name)
    prompt = prompt.format(
        user_question=question,
        table_metadata_string=pruned_metadata_str,
        instructions=instructions,
    )
    return prompt
