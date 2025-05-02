import re

from datasets import Dataset, load_dataset

from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.utils import get_logger

from .base import BaseDataset


def _parse(item, prompt_mode):
    item['expert'] = item['Bio_Category']
    item['start'] = chr(65)
    item['end'] = chr(65 + len(item.get('choices', {'label': []})['label']) -
                      1)
    item['prompt_mode'] = prompt_mode
    return item


@LOAD_DATASET.register_module()
class CARDBiomedBenchDataset(BaseDataset):

    @staticmethod
    def load(path: str, prompt_mode: str, **kwargs):
        data_files = {'test': 'data/CARDBiomedBench.csv'}
        dataset = load_dataset(path, data_files=data_files, split='test')
        # dataset = dataset.select(range(200))
        if prompt_mode == 'zero-shot':
            dataset = dataset.map(lambda item: _parse(item, prompt_mode),
                                  load_from_cache_file=False)
        elif prompt_mode == 'few-shot':
            pass  # TODO: Implement few-shot prompt
        return dataset


def _generic_llmjudge_postprocess(judgement: str):
    match = re.search(r'(A|B)', judgement)
    grade_letter = (match.group(0) if match else 'B'
                    )  # Default to "INCORRECT" if no match
    return grade_letter


def CARDBiomedBench_llmjudge_postprocess(
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

    total_correct = 0
    total_count = 0

    for k, v in output.items():
        idx = int(k)  # Convert key to integer for indexing
        original_responses.append(v['prediction'])
        processed_judge = _generic_llmjudge_postprocess(v['prediction'])

        sample = original_dataset[idx]
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

            # Add to details
            details.append({
                'id': k,
                'question': sample['question'],
                'prediction': sample['prediction'],
                'origin_prompt': v['origin_prompt'],
                'llm_judge': processed_judge,
                'gold': gold,
                'is_correct': is_correct,
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
    return results
