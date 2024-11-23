
import json
import os
from datasets import Dataset
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path
from ..base import BaseDataset
from opencompass.datasets.korbench.korbench_utils import evaluate_responses, load_json_or_jsonl, find_file, load_yaml

import yaml
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS

@LOAD_DATASET.register_module()
class korbenchsingle3shotDataset(BaseDataset):
    """
    Dataset loader for the  task in KOR-Bench.
    """

    @staticmethod
    def load(path, category):
        """
        Load the  dataset using shared .
        """
        base_path = get_data_path(path)
        rule_file = find_file(base_path, os.path.join(category, "rule"))
        sample_file = find_file(base_path, os.path.join(category, "sample"))
        few_shot_file = find_file(base_path, os.path.join(category, "three-shot"))
        
        # Load data
        rules = load_json_or_jsonl(rule_file) or []
        samples = load_json_or_jsonl(sample_file) or []

        # Load the prompt template
        template_path = os.path.join(os.path.dirname(__file__), "korbench_dataset_config/prompt/3_shot.yaml")
        few_shot = load_json_or_jsonl(few_shot_file) or []
        print(f"template_path: {template_path}")
        try:
            template = load_yaml(template_path)
        except FileNotFoundError:
            print(f"[ERROR] Missing prompt template: {template_path}")
            return Dataset.from_list([])

        # Process data
        data = []
        for sample in samples:
            rule_id = sample["rule_id"]
            rule = next((r for r in rules if r["idx"] == rule_id), None)
            few_shot_qa = [item for fs in few_shot if fs["rule_id"] == rule_id for item in [fs["question"], fs["answer"]]]
            if not rule:
                print(f"[WARNING] Rule ID {sample['rule_id']} not found for sample {sample}. Skipping...")
                continue

            prompt_key = f"{category}_prompt_format"
            prompt = template[prompt_key][0].format(rule["rule_content"], *few_shot_qa, sample["question"])

            # Add processed item
            data.append({
                "rule_content": rule["rule_content"],
                "question": sample["question"],
                "answer": sample["answer"],
                "prompt": prompt,
                "rule_id": rule["idx"],
                "mode": "3_shot",
                "category": category,
            })

        print(f"Loaded {len(data)} samples for the  task.")
        return Dataset.from_list(data)

@ICL_EVALUATORS.register_module()
class korbenchsingle3shotEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()

    def score(self, predictions, references, test_set):
        """
        Evaluate predictions for the  task.
        """
        dataset_scores = {}
        data = {}
        count = 0

        for i in range(len(predictions)):
            if test_set[i]["mode"] == "3_shot":
                data[count] = {
                    "prediction": predictions[i],
                    "gold": references[i],
                    "rule_id": test_set[i]["rule_id"],
                    "category": test_set[i]["category"],
                }
                count += 1
                

        if data:
            evaluation_results = evaluate_responses(data, "3_shot")
            correct_count = sum(res["is_correct"] for res in evaluation_results)
            accuracy = (correct_count / len(evaluation_results)) * 100 if evaluation_results else 0
            dataset_scores["accuracy"] = accuracy
        else:
            raise ValueError("3_shot data is empty")

        return dataset_scores
