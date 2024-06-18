import re
from typing import Dict, List

reserved_keywords = [
    "abs",
    "all",
    "and",
    "any",
    "avg",
    "as",
    "at",
    "asc",
    "bit",
    "by",
    "day",
    "dec",
    "do",
    "div",
    "end",
    "for",
    "go",
    "in",
    "is",
    "not",
    "or",
    "to",
]

def get_table_names(md: str) -> List[str]:
    """
    Given a string of metadata formatted as a series of
    CREATE TABLE statements, return a list of table names in the same order as 
    they appear in the metadata.
    """
    table_names = []
    if "CREATE TABLE" not in md:
        return table_names
    for table_md_str in md.split(");"):
        if "CREATE TABLE " not in table_md_str:
            continue
        header = table_md_str.split("(", 1)[0]
        table_name = header.split("CREATE TABLE ", 1)[1].strip()
        table_names.append(table_name)
    return table_names


def generate_aliases_dict(
    table_names: List, reserved_keywords: List[str] = reserved_keywords
) -> Dict[str, str]:
    """
    Generate aliases for table names as a dictionary mapping of table names to aliases
    Aliases should always be in lower case
    """
    aliases = {}
    for original_table_name in table_names:
        if "." in original_table_name:
            table_name = original_table_name.rsplit(".", 1)[-1]
        else:
            table_name = original_table_name
        if "_" in table_name:
            # get the first letter of each subword delimited by "_"
            alias = "".join([word[0] for word in table_name.split("_")]).lower()
        else:
            # if camelCase, get the first letter of each subword
            # otherwise defaults to just getting the 1st letter of the table_name
            temp_table_name = table_name[0].upper() + table_name[1:]
            alias = "".join(
                [char for char in temp_table_name if char.isupper()]
            ).lower()
            # append ending numbers to alias if table_name ends with digits
            m = re.match(r".*(\d+)$", table_name)
            if m:
                alias += m.group(1)
        if alias in aliases.values() or alias in reserved_keywords:
            alias = table_name[:2].lower()
        if alias in aliases.values() or alias in reserved_keywords:
            alias = table_name[:3].lower()
        num = 2
        while alias in aliases.values() or alias in reserved_keywords:
            alias = table_name[0].lower() + str(num)
            num += 1

        aliases[original_table_name] = alias
    return aliases


def mk_alias_str(table_aliases: Dict[str, str]) -> str:
    """
    Given a dictionary of table names to aliases, return a string of aliases in the form:
    -- table1 AS t1
    -- table2 AS t2
    """
    aliases_str = ""
    for table_name, alias in table_aliases.items():
        aliases_str += f"-- {table_name} AS {alias}\n"
    return aliases_str


def generate_aliases(
    table_names: List, reserved_keywords: List[str] = reserved_keywords
) -> str:
    """
    Generate aliases for table names in a comment str form, eg
    -- table1 AS t1
    -- table2 AS t2
    """
    aliases = generate_aliases_dict(table_names, reserved_keywords)
    return mk_alias_str(aliases)