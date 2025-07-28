import json
import re

from datasets import Dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class ChemExamDataset(BaseDataset):

    @staticmethod
    def load(path: str):

        path = get_data_path(path)

        with open(path, 'r', encoding='utf-8') as f:
            lines = [json.loads(line) for line in f]

        reformat_data = []
        for line in lines:
            prompt = line['question']
            output = ''
            qa_type = []
            if 'sub_question' in line:
                for i, sub_q in enumerate(line['sub_question']):
                    prompt += f"\nQ{i + 1}: {sub_q['question']}"
                    if 'sub_question' in sub_q:
                        for j, sub_sub_q in enumerate(sub_q['sub_question']):
                            prompt += (f'\nQ{i + 1}.{j + 1}: '
                                       f"{sub_sub_q['question']}")
                            output += (f'\nA{i + 1}.{j + 1}: '
                                       f"{sub_sub_q['answer']}")
                            qa_type.append(sub_sub_q['qa_type'])
                    else:
                        output += f"\nA{i + 1}: {sub_q['answer']}"
                        qa_type.append(sub_q['qa_type'])
            else:
                output = line['answer']
                qa_type.append(line['qa_type'])
            reformat_data.append({
                'id': line['id'],
                'prompt': prompt,
                'output': output,
                'has_img': line['has_img'],
                'qa_type': [str(item) for item in qa_type],
            })

        dataset = Dataset.from_list(reformat_data)
        return dataset


def chem_exam_score_llmjudge_postprocess(output, output_path, dataset):
    origin_dataset = []
    for item in dataset.reader.dataset['test']:
        origin_dataset.append(item)

    pattern = re.compile(
        r'(?:<NUMBER>\s*|\\boxed\{)\s*(-?\d*\.?\d+)\s*(?:</NUMBER>|\})')
    details = []
    for k, v in output.items():
        idx = int(k)

        # Get original item from dataset
        sample = origin_dataset[idx]
        print(f'Processing item {idx}: {sample}')

        # Extract the prediction from the output
        prediction = v['prediction']
        matches = pattern.findall(prediction)
        if not matches:
            score = 0
        else:
            score = [float(match) for match in matches][-1]
        details.append({
            'id': k,
            'question': sample['prompt'],
            'origin_prompt': v['origin_prompt'],
            'prediction': prediction,
            'gold': sample['output'],
            'score': score,
            # 'qa_type': sample['qa_type'],
            'has_img': sample['has_img'],
        })

    final_score = round(
        100 * sum(item['score'] for item in details) / len(details), 2)
    results = {
        'final_score': final_score,
        'details': details,
    }
    return results
