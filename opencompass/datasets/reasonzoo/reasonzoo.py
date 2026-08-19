import os

from datasets import Dataset

from opencompass.datasets.reasonzoo.reasonzoo_utils import (evaluate_responses,
                                                            find_file,
                                                            load_json_or_jsonl,
                                                            load_yaml)
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class reasonzooDataset(BaseDataset):
    """Dataset loader for the  task in reasonzoo."""

    @staticmethod
    def load(path, prompt_mode, category, **kwargs):
        """Load the  dataset using shared ."""
        base_path = get_data_path(path)
        # The actual data files are in a nested 'data' directory
        data_path = os.path.join(base_path, 'data')
        rule_file = None
        sample_file = None
        if '0_shot' in prompt_mode:
            rule_file = find_file(data_path, os.path.join(category, 'rule'))
            sample_file = find_file(data_path,
                                    os.path.join(category, 'sample'))
        else:
            raise ValueError(f'Unsupported prompt_mode: {prompt_mode}')
        # Load data
        if prompt_mode == '0_shot':
            rules = load_json_or_jsonl(rule_file) or []
            samples = load_json_or_jsonl(sample_file) or []
        template_path = None
        if prompt_mode == '0_shot':
            template_path = os.path.join(
                os.path.dirname(__file__),
                'reasonzoo_dataset_config/prompt/0_shot.yaml')
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


@ICL_EVALUATORS.register_module()
class reasonzooEvaluator(BaseEvaluator):

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
        }

        # Evaluate the single sample
        data = {0: entry}

        # Evaluate based on different prompt_mode
        if prompt_mode == '0_shot':
            evaluation_results = evaluate_responses(data, '0_shot')
        else:
            raise ValueError(f'Unsupported prompt_mode: {prompt_mode}')

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
