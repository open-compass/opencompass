import os
import json
from datasets import Dataset
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path
import yaml
from ..base import BaseDataset
from opencompass.datasets.korbench.korbench_utils import evaluate_responses
def load_json_or_jsonl(file_path):
    """
    Load data from a JSON or JSONL file.
    """
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r', encoding='utf-8') as file:
        if file_path.endswith('.json'):
            return json.load(file)
        elif file_path.endswith('.jsonl'):
            return [json.loads(line) for line in file]
    return None

def find_file(base_path, sub_path, extensions=('json', 'jsonl')):
    """
    Find the first available file with given extensions.
    """
    for ext in extensions:
        file_path = os.path.join(base_path, f"{sub_path}.{ext}")
        if os.path.exists(file_path):
            return file_path
    return None

def load_yaml(yaml_path):
    """
    Load a YAML file.
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")
    with open(yaml_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

@LOAD_DATASET.register_module()
class korbenchlogicDataset(BaseDataset):
    """
    Dataset loader for the logic task in KOR-Bench.
    """

    @staticmethod
    def load(path):
        """
        Load the logic dataset using shared logic.
        """
        # Resolve the base path
        base_path = get_data_path(path)
        fall_back_path = f"{os.getenv('BASE_PATH')}/data/korbench"
        # Find rule and sample files
        rule_file = find_file(base_path, os.path.join("logic", "rule"))
        sample_file = find_file(base_path, os.path.join("logic", "sample"))

        if not rule_file or not sample_file:
            rule_file = find_file(fall_back_path, os.path.join("logic", "rule"))
            sample_file = find_file(fall_back_path, os.path.join("logic", "sample"))


        # Load data
        rules = load_json_or_jsonl(rule_file) or []
        samples = load_json_or_jsonl(sample_file) or []

        # Load the prompt template
        template_path = f"{os.getenv('BASE_PATH')}/data/korbench/config/prompt/zero-shot.yaml"
        try:
            template = load_yaml(template_path)
        except FileNotFoundError:
            print(f"[ERROR] Missing prompt template: {template_path}")
            return Dataset.from_list([])

        # Process data
        data = []
        for sample in samples:
            rule = next((r for r in rules if r["idx"] == sample["rule_id"]), None)
            if not rule:
                print(f"[WARNING] Rule ID {sample['rule_id']} not found for sample {sample}. Skipping...")
                continue

            # Generate prompt using the template
            prompt = template['logic_prompt_format'][0].format(rule["rule_content"], sample["question"])

            # Add processed item
            data.append({
                "rule_content": rule["rule_content"],
                "question": sample["question"],
                "answer": sample["answer"],
                "prompt": prompt,
            })

        print(f"Loaded {len(data)} samples for the logic task.")
        return Dataset.from_list(data)

from datasets import Dataset, DatasetDict
from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.utils import get_data_path
from opencompass.openicl.utils import evaluate_response_vs_answer
from ..base import BaseDataset
from opencompass.utils import get_logger
from opencompass.registry import ICL_EVALUATORS

@ICL_EVALUATORS.register_module()
class korbenchlogicEvaluator(BaseEvaluator):
    def __init__(self, metadata_file=None, output_folder=None):
        super().__init__()
        self.metadata_file = metadata_file or f"{os.getenv('BASE_PATH')}/outputs/metadata/metadata.json"
        self.output_folder = output_folder or f"{os.getenv('BASE_PATH')}/evaluation_results"

    def load_metadata(self):
        """
        Load metadata to get prediction file paths.
        """
        if not os.path.exists(self.metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")
        with open(self.metadata_file, "r") as f:
            return json.load(f)

    def score(self):
        """
        Evaluate predictions for the Logic task.
        """
        metadata = self.load_metadata()
        os.makedirs(self.output_folder, exist_ok=True)
        dataset_scores = {}

        for entry in metadata:

            output_path = entry["output_path"]
            if not os.path.exists(output_path):
                print(f"[WARNING] Prediction file not found: {output_path}")
                continue

            with open(output_path, "r") as f:
                data = json.load(f)

            evaluation_results = evaluate_responses(data, "logic", "zero-shot")
            correct_count = sum(res["is_correct"] for res in evaluation_results)
            accuracy = (correct_count / len(evaluation_results)) * 100 if evaluation_results else 0

            dataset_scores["logic"] = accuracy

        self._save_results(dataset_scores)
        return dataset_scores

    def _save_results(self, dataset_scores):
        json_output_path = os.path.join(self.output_folder, "logic_scores.json")
        with open(json_output_path, "w", encoding="utf-8") as json_file:
            json.dump(dataset_scores, json_file, ensure_ascii=False, indent=4)