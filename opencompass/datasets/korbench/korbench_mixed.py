import os

from datasets import Dataset

from opencompass.datasets.korbench.korbench_utils import (
    evaluate_responses, find_file, load_json_or_jsonl,
    load_json_or_jsonl_with_idx, load_yaml)
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class korbenchmixedDataset(BaseDataset):
    """Dataset loader for the mixed mode task in KOR-Bench."""

    @staticmethod
    def load(path, mode):
        """Load the mixed mode dataset."""
        base_path = get_data_path(path)
        mixed_file = find_file(base_path, os.path.join('mixed', mode))

        # Load data
        mixed_data = load_json_or_jsonl(mixed_file) or []

        # Load the prompt template
        template_path = os.path.join(
            os.path.dirname(__file__),
            'korbench_dataset_config/prompt/mixed.yaml')
        try:
            template = load_yaml(template_path)
        except FileNotFoundError:
            print(f'[ERROR] Missing prompt template: {template_path}')
            return Dataset.from_list([])

        # Process data
        data = []
        for item in mixed_data:
            rule_list = item['rule_list']
            question_list = item['question_list']
            rule_content_list = []
            question_content_list = []

            # Fetch rules and questions
            for rule in rule_list:
                category, rule_idx = rule.rsplit('_', 1)
                rule_content = load_json_or_jsonl_with_idx(base_path,
                                                           os.path.join(
                                                               category,
                                                               'rule'),
                                                           idx=rule_idx)
                rule_content_list.append(rule_content['rule_content'])

            for question in question_list:
                category, question_idx = question.rsplit('_', 1)
                question_content = load_json_or_jsonl_with_idx(
                    base_path,
                    os.path.join(category, 'sample'),
                    idx=question_idx)
                question_content_list.append(question_content['question'])

            # Prepare prompt
            rules_str = '\n'.join(
                f'Rule {i+1}: {content}'
                for i, content in enumerate(rule_content_list))
            questions_str = '\n'.join(
                f'Question {i+1}: {content}'
                for i, content in enumerate(question_content_list))
            prompt_format = [rules_str, questions_str]
            prompt = template['prompt_format'][0].format(*prompt_format)

            # Add processed item
            data.append({
                'rule_list': rule_list,
                'question_list': question_list,
                'prompt': prompt,
                'mode': mode,
                'answer': '',
                'base_path': base_path,
            })

        print(f'Loaded {len(data)} samples for the mixed mode task.')
        return Dataset.from_list(data)


@ICL_EVALUATORS.register_module()
class korbenchmixedEvaluator(BaseEvaluator):

    def __init__(self):
        super().__init__()

    def score(self, predictions, references, test_set):
        """Evaluate predictions for the mixed mode task."""
        dataset_scores = {}
        data = {}
        count = 0

        for i in range(len(predictions)):
            if test_set[i]['mode'] in ['Multi-Q', 'Multi-R', 'Multi-RQ']:
                data[count] = {
                    'prediction': predictions[i],
                    'gold': references[i],
                    'rule_list': test_set[i]['rule_list'],
                    'question_list': test_set[i]['question_list'],
                }
                count += 1

        if data:
            evaluation_results = evaluate_responses(data, 'mixed',
                                                    test_set[0]['base_path'])
            correct_count = sum(res['is_correct']
                                for res in evaluation_results)
            accuracy = (correct_count / len(evaluation_results)
                        ) * 100 if evaluation_results else 0
            dataset_scores['accuracy'] = accuracy
        else:
            raise ValueError('Mixed mode data is empty')

        return dataset_scores
