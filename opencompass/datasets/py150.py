from datasets import Dataset
import json
import re
from .base import BaseDataset

from opencompass.registry import LOAD_DATASET

def py150_post_process(code):
    code = code.replace("<NUM_LIT>", "0").replace("<STR_LIT>", "").replace("<CHAR_LIT>", "")
    pattern = re.compile(r"<(STR|NUM|CHAR)_LIT:(.*?)>", re.S)
    lits = re.findall(pattern, code)
    for lit in lits:
        code = code.replace(f"<{lit[0]}_LIT:{lit[1]}>", lit[1])
    code = json.loads(code)
    code['input'] = code['input'].replace("<s>", "").split("<EOL>")
    for code_line in code['input']:
        code_line = code_line.strip() 
    code['input'] = "\n".join(code['input'])
    code.pop("id", None)
    return code

@LOAD_DATASET.register_module()
class Py150Dataset(BaseDataset):
    @staticmethod
    def load(path):
        lines = open(path, "r").readlines()
        rows = []
        for line in lines:
            row = py150_post_process(line)
            rows.append(row)
        return Dataset.from_list(rows)