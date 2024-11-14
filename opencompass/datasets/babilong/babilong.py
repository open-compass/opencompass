# flake8: noqa: F401, E501
import json
import os

from datasets import Dataset

from opencompass.datasets.babilong.babilong_utils import compare_answers
from opencompass.datasets.babilong.prompts import (DEFAULT_PROMPTS,
                                                   DEFAULT_TEMPLATE,
                                                   get_formatted_input)
from opencompass.datasets.base import BaseDataset
from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path


@LOAD_DATASET.register_module()
class BabiLongDataset(BaseDataset):

    @staticmethod
    def load(
        path,
        task,
        split_name,
        use_instruction=True,
        use_examples=True,
        use_post_prompt=True,
    ) -> Dataset:

        assert task in [
            'qa1',
            'qa2',
            'qa3',
            'qa4',
            'qa5',
            'qa6',
            'qa7',
            'qa8',
            'qa9',
            'qa10',
        ], f"Task must be in ['qa1', 'qa2', 'qa3', 'qa4', 'qa5', 'qa6', 'qa7', 'qa8', 'qa9', 'qa10']"
        assert split_name in [
            '0k',
            '1k',
            '2k',
            '4k',
            '8k',
            '16k',
            '32k',
            '64k',
            '128k',
            '256k',
            '512k',
            '1m',
        ], f"Split name must be in ['0k', '1k', '2k', '4k', '8k', '16k', '32k', '64k', '128k', '256k', '512k', '1m']"

        # configure the prompt
        prompt_cfg = {
            'instruction':
            (DEFAULT_PROMPTS[task]['instruction'] if use_instruction else ''),
            'examples':
            (DEFAULT_PROMPTS[task]['examples'] if use_examples else ''),
            'post_prompt':
            (DEFAULT_PROMPTS[task]['post_prompt'] if use_post_prompt else ''),
            'template':
            DEFAULT_TEMPLATE,
        }

        path = get_data_path(path)
        file = os.path.join(path, task, f'{split_name}.json')

        with open(file, 'r') as f:
            task_data = json.load(f)

        data = []
        for sample in task_data:
            tmp_data = {'prompt': [], 'answer': []}
            target = sample['target']
            context = sample['input']
            question = sample['question']

            input_text = get_formatted_input(
                context,
                question,
                prompt_cfg['examples'],
                prompt_cfg['instruction'],
                prompt_cfg['post_prompt'],
                template=DEFAULT_TEMPLATE,
            )

            tmp_data['prompt'].append(input_text)
            tmp_data['answer'].append(target)
            data.append(tmp_data)
        return Dataset.from_list(data)


class BabiLongEvaluator(BaseEvaluator):

    def score(self, predictions, gold):
        assert len(predictions) == len(gold)
        score = (sum([
            compare_answers(str(ref[0]), pred)
            for pred, ref in zip(predictions, gold)
        ]) / len(predictions) * 100)
        result = {'score': round(score, 2)}
        return result
