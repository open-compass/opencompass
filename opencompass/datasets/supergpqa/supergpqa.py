import os
import re

from datasets import Dataset, load_dataset

from opencompass.datasets.supergpqa.supergpqa_eval import (
    extract_option_content, extract_option_labels)
from opencompass.datasets.supergpqa.supergpqa_utils import load_yaml
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_logger

from ..base import BaseDataset


def _parse(item, template, prompt_mode):
    prompt_format = [
        item['question'] + '\n' + '\n'.join([
            f'{chr(65+i)}) {option}'
            for i, option in enumerate(item['options'])
        ])
    ]
    item['infer_prompt'] = template['prompt_format'][0].format(*prompt_format)
    item['prompt_mode'] = prompt_mode
    return item


@LOAD_DATASET.register_module()
class SuperGPQADataset(BaseDataset):

    @staticmethod
    def load(path: str,
             prompt_mode: str,
             discipline: str = None,
             field: str = None,
             subfield: str = None,
             **kwargs):
        dataset = load_dataset(path, split='train')

        if discipline is not None:
            dataset = dataset.filter(lambda x: x['discipline'] == discipline)
        if field is not None:
            dataset = dataset.filter(lambda x: x['field'] == field)
        if subfield is not None:
            dataset = dataset.filter(lambda x: x['subfield'] == subfield)

        # get prompt template
        template_path = None
        if prompt_mode == 'zero-shot':
            template_path = os.path.join(
                os.path.dirname(__file__),
                'supergpqa_dataset_config/prompt/zero-shot.yaml',
            )
        elif prompt_mode == 'five-shot':
            template_path = os.path.join(
                os.path.dirname(__file__),
                'supergpqa_dataset_config/prompt/five-shot.yaml',
            )
        try:
            template = load_yaml(template_path)
        except FileNotFoundError:
            print(f'[ERROR] Missing prompt template: {template_path}')
            return Dataset.from_list([])

        dataset = dataset.map(lambda item: _parse(item, template, prompt_mode))
        return dataset


@ICL_EVALUATORS.register_module()
class SuperGPQAEvaluator(BaseEvaluator):

    def __init__(self):
        super().__init__()

    def score(self, predictions, references, test_set):
        mode = test_set[0]['prompt_mode']
        acc = 0
        count = 0
        err = 0
        miss = 0
        acc_difficulty = {'hard': 0, 'middle': 0, 'easy': 0}
        count_difficulty = {'hard': 0, 'middle': 0, 'easy': 0}
        stats = {'discipline': {}, 'field': {}, 'subfield': {}}
        details = []
        for i, sample in enumerate(test_set):
            sample['pred'] = prediction = predictions[i]
            gold = references[i]
            if mode == 'zero-shot':
                predict = extract_option_labels(prediction, 'ABCDEFGHIJ')
                if predict is None:
                    predict = extract_option_content(prediction,
                                                     sample['options'])
                    predict = (chr(sample['options'].index(predict) +
                                   65) if predict else None)
                sample['extracted_answer'] = predict
            elif mode == 'five-shot':
                response = prediction.split('Question:')[0]
                predict = extract_option_labels(response, 'ABCDEFGHIJ')
                if predict is None:
                    predict = extract_option_content(response,
                                                     sample['options'])
                    predict = (chr(sample['options'].index(predict) +
                                   65) if predict else None)
                if predict is None:
                    predict = extract_option_labels(prediction, 'ABCDEFGHIJ')
                    if predict is None:
                        predict = extract_option_content(
                            prediction, sample['options'])
                        predict = (chr(sample['options'].index(predict) +
                                       65) if predict else None)
                sample['extracted_answer'] = predict

            discipline = sample.get('discipline', 'unknown')
            field = sample.get('field', 'unknown')
            subfield = sample.get('subfield', 'unknown')
            difficulty = sample.get('difficulty', 'unknown')

            for level, key in [
                ('discipline', discipline),
                    # ('field', f"{discipline}/{field}"),
                    # ('subfield', f"{discipline}/{field}/{subfield}"),
            ]:
                if key not in stats[level]:
                    stats[level][key] = {
                        'correct': 0,
                        'total': 0,
                        'miss': 0,
                        'error': 0,
                        'discipline': discipline,
                        'field': field,
                        'subfield': subfield,
                        'difficulty': {
                            'easy': {
                                'correct': 0,
                                'total': 0
                            },
                            'middle': {
                                'correct': 0,
                                'total': 0
                            },
                            'hard': {
                                'correct': 0,
                                'total': 0
                            },
                        },
                    }

                stats[level][key]['total'] += 1
                stats[level][key]['difficulty'][difficulty]['total'] += 1

                answer_letter = sample['answer_letter']
                assert answer_letter == gold
                if predict and answer_letter == predict:
                    acc += 1
                    acc_difficulty[difficulty] += 1
                    sample['status'] = 'correct'
                    stats[level][key]['correct'] += 1
                    stats[level][key]['difficulty'][difficulty]['correct'] += 1
                elif predict is None or predict == '':
                    miss += 1
                    sample['status'] = 'miss'
                    stats[level][key]['miss'] += 1
                elif predict == 'error':
                    err += 1
                    sample['status'] = 'error'
                    stats[level][key]['error'] += 1
                else:
                    sample['status'] = 'incorrect'
                count += 1
                count_difficulty[difficulty] += 1
                details.append({
                    'pred':
                    sample['pred'],
                    'answer':
                    sample['answer'],
                    'parsed_answer':
                    sample['extracted_answer'],
                    'correct':
                    True if sample['status'] == 'correct' else False,
                })

        return {
            'accuracy':
            acc / count if count > 0 else 0,
            'error_rate':
            err / count if count > 0 else 0,
            'miss_rate':
            miss / count if count > 0 else 0,
            'hard_accuracy':
            (acc_difficulty['hard'] /
             count_difficulty['hard'] if count_difficulty['hard'] > 0 else 0),
            'middle_accuracy':
            (acc_difficulty['middle'] / count_difficulty['middle']
             if count_difficulty['middle'] > 0 else 0),
            'easy_accuracy':
            (acc_difficulty['easy'] /
             count_difficulty['easy'] if count_difficulty['easy'] > 0 else 0),
            'details':
            details,
        }


def _generic_llmjudge_postprocess(judgement: str):
    match = re.search(r'(A|B)', judgement)
    grade_letter = (match.group(0) if match else 'B'
                    )  # Default to "INCORRECT" if no match
    return grade_letter


def supergpqa_llmjudge_postprocess(
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
    stats = {'discipline': {}, 'field': {}, 'subfield': {}}

    total_correct = 0
    total_count = 0

    # Process each sample
    for k, v in output.items():
        idx = int(k)  # Convert key to integer for indexing
        original_responses.append(v['prediction'])
        processed_judge = _generic_llmjudge_postprocess(v['prediction'])

        # Get category information from the dataset
        sample = original_dataset[idx]
        discipline = sample.get('discipline', 'unknown')
        field = sample.get('field', 'unknown')
        subfield = sample.get('subfield', 'unknown')

        # Initialize category stats if not exists
        for level, key in [
            ('discipline', discipline),
            ('field', f'{discipline}/{field}'),
            ('subfield', f'{discipline}/{field}/{subfield}'),
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
                    ('discipline', discipline),
                    ('field', f'{discipline}/{field}'),
                    ('subfield', f'{discipline}/{field}/{subfield}'),
                ]:
                    stats[level][key]['correct'] += 1

            # Update category totals
            for level, key in [
                ('discipline', discipline),
                ('field', f'{discipline}/{field}'),
                ('subfield', f'{discipline}/{field}/{subfield}'),
            ]:
                stats[level][key]['total'] += 1
            # Add to details
            details.append({
                'id': k,
                'question': sample['question'],
                'options': sample['options'],
                'origin_prompt': v['origin_prompt'],
                'prediction': processed_judge,  # llm_judge
                'gold': gold,
                'is_correct': is_correct,
                'discipline': discipline,
                'field': field,
                'subfield': subfield,
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
                flat_key = f'SuperGPQA-{level}'
                if level == 'discipline':
                    flat_key = f'SuperGPQA-{key}'
                elif level == 'field':
                    discipline, field = key.split('/')
                    flat_key = f'SuperGPQA-{discipline}-{field}'
                elif level == 'subfield':
                    discipline, field, subfield = key.split('/')
                    flat_key = f'SuperGPQA-{discipline}-{field}-{subfield}'

                # Add to results
                results[flat_key] = accuracy

    return results
