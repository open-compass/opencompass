from datasets import load_dataset

from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET
from opencompass.utils.text_postprocessors import first_option_postprocess

from .base import BaseDataset


def _parse(item):
    item['start'] = chr(65)
    item['end'] = chr(65 + len(item.get('options', [])) - 1)
    new_options = []
    choices = ''
    for i in range(len(item['options'])):
        new_options.append(item['options'][i].split(': ')[-1])
        choices += chr(65 +
                       i) + '. ' + item['options'][i].split(': ')[-1] + '\n'
    item['question'] = (f'\nQuestion: {item["question"]}\n'
                        f'Answer Choices: \n{choices}')
    item['options'] = new_options
    item['label'] = chr(65 + int(item['answer'].split(' ')[-1]) -
                        1)  # Index from 1 in answer
    return item


@LOAD_DATASET.register_module()
class ProteinLMBenchDataset(BaseDataset):

    @staticmethod
    def load(path: str, **kwargs):
        dataset = load_dataset(path, 'evaluation', split='train')
        dataset = dataset.map(lambda item: _parse(item))

        return dataset


class ProteinLMBenchEvaluator(BaseEvaluator):

    def score(self, predictions, references, test_set):
        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}
        correct = 0
        count = 0
        details = []
        for idx, (prediction,
                  reference) in enumerate(zip(predictions, references)):
            options = ''.join(
                [chr(65 + i) for i in range(len(test_set['options'][idx]))])
            predict = first_option_postprocess(prediction, options)
            detail = {'pred': predict, 'answer': reference, 'correct': False}
            count += 1
            if predict == reference:
                correct += 1
                detail['correct'] = True
            details.append(detail)
        result = {'accuracy': 100 * correct / count, 'details': details}
        return result
