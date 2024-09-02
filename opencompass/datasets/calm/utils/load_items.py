import json
from pathlib import Path


def load_query_instances(path):
    """Loads query instances from a JSON file.

    Args:
        path (str or Path): The path to the JSON file.

    Returns:
        list: A list of query instances loaded from the JSON file.
    """
    if isinstance(path, str):
        path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        item_list = [json.loads(line) for line in f.readlines()]
    return item_list
