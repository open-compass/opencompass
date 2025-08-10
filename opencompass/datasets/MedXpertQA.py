import re

from datasets import Dataset, load_dataset

from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.utils import get_logger

from .base import BaseDataset


def _parse(item, prompt_mode):
    item['start'] = chr(65)
    item['end'] = chr(65 + len(item.get('options', [])) - 1)
    item['prompt_mode'] = prompt_mode
    return item


@LOAD_DATASET.register_module()
class MedXpertQADataset(BaseDataset):

    @staticmethod
    def load(path: str, prompt_mode: str, **kwargs):
        dataset = load_dataset(path, 'Text', split='test')
        # dataset = load_dataset(path, 'Text', split='dev')

        if prompt_mode == 'zero-shot':
            dataset = dataset.map(lambda item: _parse(item, prompt_mode))
        elif prompt_mode == 'few-shot':
            pass  # TODO: Implement few-shot prompt

        return dataset


class MedXpertQAEvaluator(BaseEvaluator):

    def score(self, predictions, references, test_set):
        method = test_set['prompt_mode'][0]

        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}
        correct = 0
        count = 0
        details = []
        for idx, (i, j) in enumerate(zip(predictions, references)):
            i = answer_cleansing(method, i, test_set['options'][idx],
                                 test_set['label'][idx])
            detail = {'pred': i, 'answer': j, 'correct': False}
            count += 1
            if i == j:
                correct += 1
                detail['correct'] = True
            details.append(detail)
        result = {'accuracy': 100 * correct / count, 'details': details}
        return result


@TEXT_POSTPROCESSORS.register_module()
def answer_cleansing(
    method: str,
    prediction: str,
    options: list,
    label: str,
) -> str:

    # Clean up unwanted phrases in the prediction
    for unwanted_phrase in [
            'I understand',
            'A through J',
            'A through E',
            'A through D',
    ]:
        prediction = prediction.replace(unwanted_phrase, '')

    options_num = len(options)
    options = [chr(65 + i) for i in range(options_num)]
    options_str = r'\b(' + '|'.join(options) + r')\b'
    prediction = re.findall(options_str, prediction)

    if len(prediction) == 0:
        prediction = []
    else:
        # If there is a "label" and its length is 1,
        # process prediction accordingly
        if len(label) == 1:
            if method == 'few-shot':
                answer_flag = True if len(prediction) > 1 else False
                # choose the first or last element based on the answer_flag
                if answer_flag:
                    prediction = [prediction[0]]
                else:
                    prediction = [prediction[-1]]
            elif method == 'zero-shot':
                # choose the first element in list
                prediction = [prediction[0]]
            else:
                raise ValueError('Method is not properly defined ...')

            # Remove trailing period if it exists
            if prediction[0] and prediction[0].endswith('.'):
                prediction[0] = prediction[0][:-1]

    return prediction[0]


def _generic_llmjudge_postprocess(judgement: str):
    match = re.search(r'(A|B)', judgement)
    grade_letter = (match.group(0) if match else 'B'
                    )  # Default to "INCORRECT" if no match
    return grade_letter


def MedXpertQA_llmjudge_postprocess(
    output: dict,
    output_path: str,
    dataset: Dataset,
) -> dict:
    # Get the original dataset
    original_dataset = dataset.reader.dataset['test']

    judged_answers = []
    original_responses = []
    references = []
    details = []

    # Initialize statistics dictionaries
    stats = {'medical_task': {}, 'body_system': {}, 'question_type': {}}

    total_correct = 0
    total_count = 0

    # Process each sample
    for k, v in output.items():
        idx = int(k)  # Convert key to integer for indexing
        original_responses.append(v['prediction'])
        processed_judge = _generic_llmjudge_postprocess(v['prediction'])

        # Get category information from the dataset
        sample = original_dataset[idx]
        medical_task = sample.get('medical_task', 'unknown')
        body_system = sample.get('body_system', 'unknown')
        question_type = sample.get('question_type', 'unknown')

        # Initialize category stats if not exists
        for level, key in [
            ('medical_task', medical_task),
            ('body_system', body_system),
            ('question_type', question_type),
        ]:
            if key not in stats[level]:
                stats[level][key] = {'correct': 0, 'total': 0}

        # Record the judgment
        if processed_judge is not None:
            judged_answers.append(processed_judge)
            try:
                gold = v['gold']
                references.append(gold)
            except KeyError:
                get_logger().warning(
                    f'No gold answer for {k}, use empty string as reference!')
                gold = ''
                references.append('')

            # Check if the answer is correct (A means correct)
            is_correct = processed_judge == 'A'
            total_count += 1

            if is_correct:
                total_correct += 1
                # Update category stats
                for level, key in [
                    ('medical_task', medical_task),
                    ('body_system', body_system),
                    ('question_type', question_type),
                ]:
                    stats[level][key]['correct'] += 1

            # Update category totals
            for level, key in [
                ('medical_task', medical_task),
                ('body_system', body_system),
                ('question_type', question_type),
            ]:
                stats[level][key]['total'] += 1
            # Add to details
            details.append({
                'id': k,
                'question': sample['question'],
                'options': sample['options'],
                'origin_prompt': v['origin_prompt'],
                'llm_judge': processed_judge,
                'gold': gold,
                'is_correct': is_correct,
                'medical_task': medical_task,
                'body_system': body_system,
                'question_type': question_type,
            })

    # Calculate overall accuracy with two decimal places
    overall_accuracy = (round(
        (total_correct / total_count * 100), 2) if total_count > 0 else 0.00)

    # Initialize results dictionary
    results = {
        'accuracy': overall_accuracy,
        'total_correct': total_correct,
        'total_count': total_count,
        'details': details,
    }

    # Calculate accuracy for each category and flatten into results
    for level in stats:
        for key, value in stats[level].items():
            if value['total'] > 0:
                # Calculate accuracy with two decimal places
                accuracy = round((value['correct'] / value['total'] * 100), 2)

                # Create a flattened key for the category
                flat_key = f'MedXpertQA-{key}'

                # Add to results
                results[flat_key] = accuracy

    return results
