import re

from datasets import Dataset, load_dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_logger

from .base import BaseDataset


@LOAD_DATASET.register_module()
class Earth_Silver_MCQDataset(BaseDataset):

    name = 'msearth_mcq'

    @staticmethod
    def load(path: str, prompt_mode: str = 'zero-shot', **kwargs):

        dataset = load_dataset(path=path, split='multiple_choice')

        dataset = dataset.map(lambda item: {
            'question': item['question'],
            'answer': item['answer']
        })

        if prompt_mode == 'zero-shot':
            return dataset
        elif prompt_mode == 'few-shot':
            raise NotImplementedError('few-shot prompt 尚未实现')
        else:
            raise ValueError(f'Unsupported prompt_mode: {prompt_mode}')


def _generic_llmjudge_postprocess(judgement: str):
    match = re.search(r'(A|B)', judgement)
    grade_letter = (match.group(0) if match else 'B'
                    )  # Default to "INCORRECT" if no match
    return grade_letter


def earth_silver_llmjudge_postprocess(
    output: dict,
    output_path: str,
    dataset: Dataset,
) -> dict:
    # Get the original dataset
    original_dataset = dataset.reader.dataset['multiple_choice']

    judged_answers = []
    original_responses = []
    references = []
    details = []

    # Initialize statistics dictionaries
    stats = {'subject': {}, 'topic': {}, 'question_type': {}}

    total_correct = 0
    total_count = 0

    # Process each sample
    for k, v in output.items():
        idx = int(k)  # Convert key to integer for indexing
        original_responses.append(v['prediction'])

        processed_judge = _generic_llmjudge_postprocess(v['prediction'])

        # Get category information from the dataset
        sample = original_dataset[idx]
        subject = sample.get('subject_name', 'unknown')
        question_type = sample.get('choice_type', 'unknown')
        topic = sample.get('topic_name', 'unknown')

        # Initialize category stats if not exists
        for level, key in [
            ('subject', subject),
            ('question_type', question_type),
            ('topic', topic),
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
                    ('subject', subject),
                    ('question_type', question_type),
                    ('topic', topic),
                ]:
                    stats[level][key]['correct'] += 1

            # Update category totals
            for level, key in [
                ('subject', subject),
                ('question_type', question_type),
                ('topic', topic),
            ]:
                stats[level][key]['total'] += 1
            # Add to details
            details.append({
                'id': k,
                'question': sample['question'],
                'origin_prompt': v['origin_prompt'],
                'llm_judge': processed_judge,
                'gold': gold,
                'is_correct': is_correct,
                'subject': subject,
                'question_type': question_type,
                'topic': topic,
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
                flat_key = f'earth-silver-{key}'

                # Add to results
                results[flat_key] = accuracy

    return results
