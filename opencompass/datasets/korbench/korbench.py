import json
import os
import re
from os import environ

from datasets import Dataset, DatasetDict
from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.utils import get_data_path
from opencompass.openicl.utils import evaluate_response_vs_answer
from ..base import BaseDataset
from opencompass.utils import get_logger
from opencompass.registry import ICL_EVALUATORS
import csv
from prettytable import PrettyTable
from opencompass.datasets.korbench.korbench_utils import evaluate_responses
import yaml
from tqdm import tqdm


# Define the fallback base path
FALLBACK_BASE_PATH = "/home/epsilon/miniforge3/my_opencompass_project/opencompass/data/korbench"


def load_yaml(yaml_path):
    """
    Load a YAML file.
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")
    with open(yaml_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

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

def load_split_data(base_path, split_name):
    """
    Load the rule and sample data for a specific split.
    """
    split_path = os.path.join(base_path, split_name)
    rule_path = find_file(split_path, "rule")
    sample_path = find_file(split_path, "sample")

    rules = load_json_or_jsonl(rule_path) if rule_path else []
    samples = load_json_or_jsonl(sample_path) if sample_path else []

    return {"rules": rules, "samples": samples}

def process_mixed_data(base_path, mode):
    """
    Load and process data for the 'mixed' split and specific mode.
    """
    mixed_path = os.path.join(base_path, "mixed")
    file_path = find_file(mixed_path, mode)
    if not file_path:
        print(f"[WARNING] Missing file for mixed mode: {mode}")
        return []

    data = load_json_or_jsonl(file_path)
    template_path = os.path.join(base_path, "config/prompt/mixed.yaml")
    template = load_yaml(template_path)
    
    processed = []
    for item in data:
        rules = "\n".join(item.get('rule_list', []))
        questions = "\n".join(item.get('question_list', []))
        item['prompt'] = template['prompt_format'][0].format(rules, questions)
        processed.append(item)
    
    return processed

def load_korbench_dataset(path):
    """
    Load the entire KOR-Bench dataset from the given base path.
    """
    splits = ["cipher", "logic", "operation", "puzzle", "counterfactual"]
    mixed_modes = ["Multi-Q", "Multi-R", "Multi-RQ", "Multi-RQ_Hard"]
    all_data = []

    # Load standard splits
    for split in splits:
        split_data = load_split_data(base_path, split)
        all_data.extend(split_data['samples'])

    # Load mixed data
    for mode in mixed_modes:
        mixed_data = process_mixed_data(base_path, mode)
        all_data.extend(mixed_data)

    return all_data

def read_yaml(config='default'):
    """
    Read a YAML file and return its content.
    """
    # Construct the YAML file path
    yaml_file = os.path.join("/home/epsilon/miniforge3/my_opencompass_project/opencompass/data/korbench/config/prompt", f"{config}.yaml")

    # Try the fallback path first
    if os.path.exists(yaml_file):
        pass
    else:
        raise FileNotFoundError(f"No YAML configuration file found for '{config}' in either 'config/prompt/' or fallback path.")

    # Load and return YAML content
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)

def read_json_or_jsonl(base_path, split='', mapping_key=None):
    """
    Read data from a JSON or JSONL file, prioritizing the fallback path.
    """
    # Construct the full path
    if split:
        base_path = os.path.join(base_path, split)
    else:
        base_path = base_path

    # Try the fallback path first
    fallback_base_path = os.path.join(FALLBACK_BASE_PATH, split)
    file_paths = [
        f'{fallback_base_path}.json',
        f'{fallback_base_path}.jsonl',
        f'{base_path}.json',
        f'{base_path}.jsonl'
    ]

    for file_path in file_paths:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                if file_path.endswith('.json'):
                    data = json.load(file)
                elif file_path.endswith('.jsonl'):
                    data = [json.loads(line) for line in file]
                if mapping_key:
                    return {item[mapping_key]: item for item in data if mapping_key in item}
                else:
                    return data

    # Log the missing file and return an empty list
    print(f"[WARNING] No JSON or JSONL file found for '{split}' in {base_path} or fallback path.")
    return []

def read_json_or_jsonl_with_idx(base_path, split='', idx=None):
    """
    Read data from a JSON/JSONL file and find an entry with a specific index, prioritizing the fallback path.
    """
    data = read_json_or_jsonl(base_path, split)
    if idx is not None:
        try:
            return next(item for item in data if str(item.get('idx')) == str(idx))
        except StopIteration:
            raise ValueError(f"No entry found for idx {idx}")
    else:
        return data

@LOAD_DATASET.register_module()
class korbenchDataset(BaseDataset):
    """
    korbenchDataset for KOR-BENCH with flexible configurations and modes.
    """

    @staticmethod
    def load(path):
        """
        Load the dataset by processing data from rule.jsonl, sample.jsonl, and configurations.
        """

        # Define split modes
        splits_modes = {
            'cipher': ['zero-shot'],
            'logic': ['zero-shot'],
            'operation': ['zero-shot'],
            'puzzle': ['zero-shot'],
            'counterfactual': ['zero-shot'],
            'mixed': ['Multi-Q', 'Multi-R', 'Multi-RQ', 'Multi-RQ_Hard']
        }

        all_processed_data = []

        # Process each split and mode
        for split, modes in splits_modes.items():
            for mode in modes:
                # Load mixed data
                if split == 'mixed' and mode in ['Multi-Q', 'Multi-R', 'Multi-RQ', 'Multi-RQ_Hard']:
                    mixed_data = load_json_or_jsonl(find_file(FALLBACK_BASE_PATH, os.path.join('mixed', mode)))
                    if not mixed_data:
                        mixed_data = load_json_or_jsonl(find_file(path, os.path.join('mixed', mode)))
                    if not mixed_data:
                        print(f"[WARNING] Missing data for mixed split '{mode}'. Skipping...")
                        continue

                    # Load the mixed YAML template
                    template = load_yaml(os.path.join(FALLBACK_BASE_PATH, "config/prompt/mixed.yaml"))

                    for item in mixed_data:
                        # Generate the prompt from rules and questions
                        rule_list = '\n'.join(item.get('rule_list', []))
                        question_list = '\n'.join(item.get('question_list', []))
                        prompt = template['prompt_format'][0].format(rule_list, question_list)

                        # Update item with processed details
                        item['prompt'] = prompt
                        item['split'] = split
                        item['mode'] = mode
                        all_processed_data.append(item)

                # Load other splits
                else:
                    # Load rules and samples
                    rules = load_json_or_jsonl(find_file(FALLBACK_BASE_PATH, os.path.join(split, 'rule')))
                    if not rules:
                        rules = load_json_or_jsonl(find_file(path, os.path.join(split, 'rule')))
                    samples = load_json_or_jsonl(find_file(FALLBACK_BASE_PATH, os.path.join(split, 'sample')))
                    if not samples:
                        samples = load_json_or_jsonl(find_file(path, os.path.join(split, 'sample')))
                    if not rules or not samples:
                        print(f"[WARNING] Missing rules or samples for split '{split}'. Skipping...")
                        continue

                    # Load YAML template for the mode
                    config = mode
                    if mode in ['self-correction', 'self-correction-with-needle']:
                        config = 'zero-shot'
                    template_path = os.path.join(FALLBACK_BASE_PATH, f"config/prompt/{config}.yaml")
                    template = load_yaml(template_path)

                    for sample in samples:
                        rule_id = sample['rule_id']
                        rule = next((r for r in rules if r.get('idx') == rule_id), None)

                        if not rule:
                            print(f"[WARNING] Rule ID '{rule_id}' not found in rules. Skipping...")
                            continue

                        # Generate the prompt
                        rule_content = rule['rule_content']
                        question = sample['question']
                        prompt_format = [rule_content, question]
                        prompt = template[f'{split}_prompt_format'][0].format(*prompt_format)

                        # Add processed data
                        sample['prompt'] = prompt
                        sample['rule_content'] = rule_content
                        sample['split'] = split
                        sample['mode'] = mode
                        all_processed_data.append(sample)

        # Return the combined dataset
        return Dataset.from_list(all_processed_data)

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS
from opencompass.datasets.korbench.korbench_utils import evaluate_responses
import os
import json

MIXED_MODES = ["Multi-Q", "Multi-R", "Multi-RQ"]

    
@ICL_EVALUATORS.register_module()
class korbenchEvaluator(BaseEvaluator):
    def __init__(self, metadata_file=None, output_folder=None, csv_file=None):
        super().__init__()
        self.metadata_file = metadata_file or '/home/epsilon/miniforge3/my_opencompass_project/opencompass/outputs/metadata/metadata.json'
        self.output_folder = output_folder or '/home/epsilon/miniforge3/my_opencompass_project/opencompass/evaluation_results'
        self.csv_file = csv_file or '/home/epsilon/miniforge3/my_opencompass_project/opencompass/evaluation_results/summary.csv'

    def load_metadata(self):
        """Load metadata to get prediction file paths."""
        if not os.path.exists(self.metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")
        with open(self.metadata_file, 'r') as f:
            return json.load(f)

    def score(self):
        """Main evaluation logic."""
        metadata = self.load_metadata()
        os.makedirs(self.output_folder, exist_ok=True)
        dataset_scores = {}

        for entry in metadata:
            output_path = entry['output_path']
            timestamp = entry['timestamp']

            # Check if the prediction file exists
            if not os.path.exists(output_path):
                print(f"[WARNING] Prediction file not found: {output_path}")
                continue

            # Load predictions from the file
            print(f"Processing predictions from {output_path}...")
            with open(output_path, 'r') as f:
                data = json.load(f)

            # Infer dataset name and mode from the filename
            filename = os.path.basename(output_path)
            parts = filename.split('_')
            question_type = parts[1]
            mode = parts[-1].replace('.json', '')
            dataset_key = f"{question_type}_{mode}"

            # Perform evaluation
            evaluation_results = evaluate_responses(data, question_type, mode)

            # Compute statistics
            correct_count = sum(result['is_correct'] for result in evaluation_results)
            count = len(evaluation_results)
            accuracy = (correct_count / count) * 100 if count > 0 else 0

            # Store results by dataset
            if dataset_key not in dataset_scores:
                dataset_scores[dataset_key] = accuracy

        # Save results
        self._save_results(dataset_scores)
        return dataset_scores

    def _save_results(self, dataset_scores):
        """Save results to JSON and CSV."""
        # Save as JSON
        json_output_path = os.path.join(self.output_folder, "dataset_scores.json")
        with open(json_output_path, "w", encoding="utf-8") as json_file:
            json.dump(dataset_scores, json_file, ensure_ascii=False, indent=4)

        