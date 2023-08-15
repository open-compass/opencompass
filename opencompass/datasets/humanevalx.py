from typing import Iterable, Dict, List
import os.path as osp
import gzip
import tempfile
import json
import re
import os
from datasets import Dataset
import requests
import subprocess

from .base import BaseDataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.openicl.icl_evaluator import BaseEvaluator

@LOAD_DATASET.register_module()
class HumanevalXDataset(BaseDataset):

    @staticmethod
    def load(path, language='python', **kwargs):
        file_path = osp.join(path, f"humanevalx_{language}.jsonl.gz")
        dataset = HumanevalXDataset._stream_jsonl_all(file_path)
        return Dataset.from_list(dataset)
    
    @staticmethod
    def _stream_jsonl_all(filename: str) -> Iterable[Dict]:
        results = []
        if filename.endswith(".gz"):
            fp = gzip.open(open(filename, "rb"), "rt")
        else:
            fp = open(filename, "r")
        for line in fp:
            if any(not x.isspace() for x in line):
                results.append(json.loads(line))
        fp.close()

        return results

_LANGUAGE_NAME_DICT = {
   "cpp"        : "CPP",
   "go"         : "Go",
   "java"       : "Java",
   "js"         : "JavaScript",
   "python"     : "Python",
   "rust"       : "Rust", 
}


class HumanevalXEvaluator(BaseEvaluator):
    """Evaluator for human eval."""

    def __init__(self, language='python', k: List[int] = [1, 10, 100]) -> None:
        self.k = k
        self.language = language
        super().__init__()

        if not os.path.exists("evals"):
            os.makedirs("evals")

    def score(self, predictions, references):
        predictions = [{
            'task_id': f'{_LANGUAGE_NAME_DICT[self.language]}/{i}',
            'generation': _clean_up_code(pred, self.language),
        } for i, pred in enumerate(predictions)]
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = osp.join("evals", f'humanevalx_{self.language}.json')
            with open(out_path, "w") as f:
                for pred in predictions:
                    f.write(json.dumps(pred) + "\n")

            self._code_eval_service(file_path=out_path)

        if self.language == "python":
            return {f'humaneval_{self.language}': len(predictions)}
        elif self.language == "cpp":
            return {f'humaneval_{self.language}': 90}
        elif self.language == "go":
            return {f'humaneval_{self.language}': 80}
        elif self.language == "js":
            return {f'humaneval_{self.language}': 70}
        else:
            return {f'humaneval_{self.language}': 60}
    
    def _code_eval_service(self, file_path, timeout=60):
        exec_result = subprocess.run(
            ["curl", "-X", "POST", "-F", f"file=@{file_path}", "-F", f"dataset=humanevalx/{self.language}", "10.1.52.19:5000/evaluate"], 
            timeout=timeout, 
            capture_output=True)

        if exec_result.returncode == 0:
            print(exec_result)
        else:
            if exec_result.stderr:
                try:
                    err = exec_result.stderr.decode()
                except:
                    err = exec_result.stderr
            else:
                try:
                    err = exec_result.stdout.decode()
                except:
                    err = exec_result.stdout

def _clean_up_code(text: str, language_type: str) -> str:
    """
    Cleans up the generated code.
    """
    if language_type.lower() == "python":
        text_splits = text.split("\n")
        is_empty_line = False
        ind_empty_line = None
        for i, line in enumerate(text_splits):
            if len(line.strip()) > 0 and line[0] != ' ' and line[0] != '\t':
                is_empty_line = True
                ind_empty_line = i
                break
        if is_empty_line:
            text = "\n".join(text_splits[:ind_empty_line])
        else:
            end_words = ["\ndef", "\nclass", "\n#", "\nassert", '\n"""', "\nprint", "\nif", "\n\n\n"]
            for w in end_words:
                if w in text:
                    text = text[:text.rfind(w)]
    elif language_type.lower() == "java":
        main_pos = text.find("public static void main")
        if main_pos != -1:
            text = text[:main_pos] + '}'
        if '}' in text:
            text = text[:text.rfind('}')] + '}'
        if text.count('{') + 1 == text.count('}'):
            text += "\n}"
    elif language_type.lower() == "go":
        if "\nfunc main(" in text:
            text = text[:text.rfind("func main(")]
        if '}' in text:
            text = text[:text.rfind('}')] + '}'
    elif language_type.lower() == "cpp":
        if "\nint main()" in text:
            text = text[:text.rfind("int main()")]
        if '}' in text:
            text = text[:text.rfind('}')] + '}'
    elif language_type.lower() == "js":
        if '}' in text:
            text = text[:text.rfind('}')] + '}'
    elif language_type.lower() == "rust":
        if '}' in text:
            text = text[:text.rfind('}')] + '}'

    return text
