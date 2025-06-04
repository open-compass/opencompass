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
class korbenchDataset(BaseDataset):
    """Dataset loader for the  task in KOR-Bench."""

    @staticmethod
    def load(path, prompt_mode, category, **kwargs):
        """Load the  dataset using shared ."""
        base_path = get_data_path(path)
        rule_file = None
        sample_file = None
        mixed_file = None
        mixed_data = None
        if '0_shot' in prompt_mode or '3_shot' in prompt_mode:
            rule_file = find_file(base_path, os.path.join(category, 'rule'))
            sample_file = find_file(base_path,
                                    os.path.join(category, 'sample'))
        elif prompt_mode == 'mixed':
            mixed_file = find_file(base_path, os.path.join('mixed', category))
            mixed_data = load_json_or_jsonl(mixed_file) or []
        else:
            raise ValueError(f'Unsupported prompt_mode: {prompt_mode}')
        three_shot_file = None
        if prompt_mode == '3_shot':
            ts_path = os.path.join(category, 'three-shot')
            three_shot_file = find_file(base_path, ts_path)
        # Load data
        if prompt_mode in ['0_shot', '3_shot']:
            rules = load_json_or_jsonl(rule_file) or []
            samples = load_json_or_jsonl(sample_file) or []
        template_path = None
        if prompt_mode == '0_shot':
            template_path = os.path.join(
                os.path.dirname(__file__),
                'korbench_dataset_config/prompt/0_shot.yaml')
        elif prompt_mode == '3_shot':
            template_path = os.path.join(
                os.path.dirname(__file__),
                'korbench_dataset_config/prompt/3_shot.yaml')
        elif prompt_mode == 'mixed':
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
        if prompt_mode == '0_shot':
            for sample in samples:
                rule_id = sample['rule_id']
                rule = next((r for r in rules if r['idx'] == rule_id), None)
                if not rule:
                    print(f"[WARNING] Rule ID {sample['rule_id']} not found."
                          'Skipping...')
                    continue
                prompt_key = f'{category}_prompt_format'
                prompt = template[prompt_key][0].format(
                    rule['rule_content'], sample['question'])

                # Add processed item
                data.append({
                    'rule_content': rule['rule_content'],
                    'question': sample['question'],
                    'answer': sample['answer'],
                    'prompt': prompt,
                    'rule_id': rule['idx'],
                    'prompt_mode': '0_shot',
                    'category': category,
                })

            return Dataset.from_list(data)

        if prompt_mode == '3_shot':
            data = []
            three_shot = load_json_or_jsonl(three_shot_file) or []
            for sample in samples:
                rule_id = sample['rule_id']
                rule = next((r for r in rules if r['idx'] == rule_id), None)
                three_shot_qa = [
                    item for fs in three_shot if fs['rule_id'] == rule_id
                    for item in [fs['question'], fs['answer']]
                ]
                if not rule:
                    print(f"[WARNING] Rule ID {sample['rule_id']} not found."
                          'Skipping...')
                    continue
                prompt_key = f'{category}_prompt_format'
                prompt = template[prompt_key][0].format(
                    rule['rule_content'], *three_shot_qa, sample['question'])
                # Add processed item
                data.append({
                    'rule_content': rule['rule_content'],
                    'question': sample['question'],
                    'answer': sample['answer'],
                    'prompt': prompt,
                    'rule_id': rule['idx'],
                    'prompt_mode': '3_shot',
                    'category': category,
                })

            return Dataset.from_list(data)

        if prompt_mode == 'mixed':
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
                    'prompt_mode': 'mixed',
                    'answer': '',
                    'base_path': base_path,
                })

            return Dataset.from_list(data)


@ICL_EVALUATORS.register_module()
class korbenchEvaluator(BaseEvaluator):

    def __init__(self):
        super().__init__()

    def sample_score(self, prediction, reference, test_item=None):
        """Evaluate a single sample.

        Args:
            prediction: The model's prediction
            reference: The reference answer
            test_item: Additional information about the test sample

        Returns:
            Dict: A dictionary containing evaluation results
        """
        if test_item is None:
            raise ValueError('Test item is required.')

        prompt_mode = test_item.get('prompt_mode')

        # Build data for a single sample
        entry = {
            'prediction': prediction,
            'gold': reference,
            'rule_id': test_item.get('rule_id', None),
            'category': test_item.get('category', None),
            'rule_list': test_item.get('rule_list', None),
            'question_list': test_item.get('question_list', None),
            'base_path': test_item.get('base_path', None),
        }

        # Evaluate the single sample
        data = {0: entry}

        # Evaluate based on different prompt_mode
        if prompt_mode == '0_shot':
            evaluation_results = evaluate_responses(data, '0_shot')
        elif prompt_mode == '3_shot':
            evaluation_results = evaluate_responses(data, '3_shot')
        elif prompt_mode in ['Multi-Q', 'Multi-R', 'Multi-RQ', 'mixed']:
            evaluation_results = evaluate_responses(data, 'mixed',
                                                    test_item.get('base_path'))
        else:
            return {
                'is_correct': False,
                'pred': prediction,
                'answer': reference
            }

        # Return evaluation results
        result = evaluation_results[0]
        result['correct'] = result['is_correct']
        result.update({'pred': prediction, 'answer': reference})
        return result

    def score(self, predictions, references, test_set):
        """Evaluate each sample using sample_score."""
        if not test_set:
            raise ValueError('Test set is empty.')

        details = []
        correct_count = 0

        # Call sample_score for each sample
        for i in range(len(predictions)):
            result = self.sample_score(predictions[i], references[i],
                                       test_set[i])
            details.append(result)
            if result.get('is_correct', False):
                correct_count += 1

        # Calculate accuracy
        accuracy = (correct_count /
                    len(predictions)) * 100 if predictions else 0

        # Return evaluation results
        return {'accuracy': accuracy, 'details': details}
