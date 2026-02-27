import json
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset, load_dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET, ICL_EVALUATORS
from opencompass.utils import get_data_path

from .base import BaseDataset


def extract_diff(response: str) -> Optional[str]:
    if response is None:
        return None
    diff_matches = []
    other_matches = []
    pattern = re.compile(r"\<([\w-]+)\>(.*?)\<\/\1\>", re.DOTALL)
    for code, match in pattern.findall(response):
        if code in {"diff", "patch"}:
            diff_matches.append(match)
        else:
            other_matches.append(match)
    pattern = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
    for code, match in pattern.findall(response):
        if code in {"diff", "patch"}:
            diff_matches.append(match)
        else:
            other_matches.append(match)
    if diff_matches:
        return diff_matches[0].strip()
    if other_matches:
        return other_matches[0].strip()
    return response.split("</s>")[0].strip()


@LOAD_DATASET.register_module()
class SWEBenchDataset(BaseDataset):

    @staticmethod
    def load(
        path: str = 'princeton-nlp/SWE-bench',
        split: str = 'test',
        local_mode: bool = False,
        max_problem_statement_length: int = 10000,
    ) -> Dataset:
        if local_mode:
            path = get_data_path(path, local_mode=local_mode)
            if os.path.exists(path):
                dataset = Dataset.load_from_disk(path)
            else:
                dataset = load_dataset(path, split=split)
        else:
            dataset = load_dataset(path, split=split)
        processed_data = []
        for item in dataset:
            problem_statement = item['problem_statement']
            if len(problem_statement) > max_problem_statement_length:
                problem_statement = problem_statement[:max_problem_statement_length] + "\n\n[Content truncated due to length...]"
            hints_text = item.get('hints_text', '') or ''
            if len(hints_text) > 2000:
                hints_text = hints_text[:2000]
            processed_item = {
                'instance_id': item['instance_id'],
                'repo': item['repo'],
                'base_commit': item['base_commit'],
                'problem_statement': problem_statement,
                'hints_text': hints_text,
                'patch': item.get('patch', ''),
            }
            processed_data.append(processed_item)
        return Dataset.from_list(processed_data)


@ICL_EVALUATORS.register_module()
class SWEBenchEvaluator(BaseEvaluator):

    def __init__(
        self,
        dataset_name: str = 'princeton-nlp/SWE-bench',
        split: str = 'test',
        max_workers: int = 4,
        timeout: int = 1800,
        run_id: str = 'opencompass_swebench_eval',
        use_modal: bool = False,
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.max_workers = max_workers
        self.timeout = timeout
        self.run_id = run_id
        self.use_modal = use_modal
        super().__init__()

    def score(self, predictions: List[str], references: List[str], test_set: Dataset) -> Dict:
        if len(predictions) != len(references):
            return {'error': 'predictions and references have different length'}
        predictions_data = []
        for i, pred in enumerate(predictions):
            model_patch = extract_diff(pred)
            instance_id = test_set[i]['instance_id']
            predictions_data.append({
                'instance_id': instance_id,
                'model_name_or_path': 'opencompass_model',
                'model_patch': model_patch if model_patch else '',
            })
        with tempfile.TemporaryDirectory() as tmp_dir:
            pred_file = os.path.join(tmp_dir, 'predictions.jsonl')
            with open(pred_file, 'w') as f:
                for pred_item in predictions_data:
                    f.write(json.dumps(pred_item) + '\n')
            try:
                import subprocess
                cmd = [
                    'python', '-m', 'swebench.harness.run_evaluation',
                    '--dataset_name', self.dataset_name,
                    '--split', self.split,
                    '--predictions_path', pred_file,
                    '--max_workers', str(self.max_workers),
                    '--run_id', self.run_id,
                    '--timeout', str(self.timeout),
                ]
                if self.use_modal:
                    cmd.extend(['--modal', 'true'])
                subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout * len(predictions) + 3600,
                )
                log_dir = Path(f'logs/run_evaluation/{self.run_id}')
                resolved_count = 0
                total_count = len(predictions_data)
                details = []
                if log_dir.exists():
                    for instance_dir in log_dir.glob('*/*'):
                        report_file = instance_dir / 'report.json'
                        if report_file.exists():
                            with open(report_file) as f:
                                report = json.load(f)
                                for inst_id, inst_result in report.items():
                                    resolved = inst_result.get('resolved', False)
                                    if resolved:
                                        resolved_count += 1
                for i, pred_item in enumerate(predictions_data):
                    inst_id = pred_item['instance_id']
                    report_found = False
                    if log_dir.exists():
                        for instance_dir in log_dir.glob(f'*/{inst_id}'):
                            report_file = instance_dir / 'report.json'
                            if report_file.exists():
                                with open(report_file) as f:
                                    report = json.load(f)
                                    if inst_id in report:
                                        resolved = report[inst_id].get('resolved', False)
                                        details.append({
                                            'instance_id': inst_id,
                                            'prediction': pred_item['model_patch'],
                                            'reference': references[i] if references else '',
                                            'resolved': resolved,
                                            'correct': resolved,
                                        })
                                        report_found = True
                                        break
                    if not report_found:
                        details.append({
                            'instance_id': inst_id,
                            'prediction': pred_item['model_patch'],
                            'reference': references[i] if references else '',
                            'resolved': False,
                            'correct': False,
                        })
                resolved_rate = (resolved_count / total_count * 100) if total_count > 0 else 0
                return {
                    'resolved_rate': resolved_rate,
                    'resolved_count': resolved_count,
                    'total_count': total_count,
                    'details': details,
                }
            except subprocess.TimeoutExpired:
                return {
                    'error': 'Evaluation timed out',
                    'resolved_rate': 0,
                    'resolved_count': 0,
                    'total_count': len(predictions_data),
                }
            except Exception as e:
                return {
                    'error': str(e),
                    'resolved_rate': 0,
                    'resolved_count': 0,
                    'total_count': len(predictions_data),
                }


def swebench_postprocess(text: str) -> str:
    return extract_diff(text) or ''
