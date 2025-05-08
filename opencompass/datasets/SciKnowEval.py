import re

from datasets import load_dataset

from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS

from .base import BaseDataset


def _parse(item, prompt_mode, discipline):
    choices = item['choices']

    item['q4'] = f'You are an expert in {discipline}.\n'
    item['q4'] += item['prompt']['default'] + '\n' + item['question'] + '\n'
    label_texts = []
    for label_meta, text_meta in zip(choices['label'], choices['text']):
        label_texts.append(f'{label_meta}. {text_meta}')
    item['q4'] += '\n'.join(label_texts)  # noqa: E501, E741, E741
    item['prompt_mode'] = prompt_mode
    return item


@LOAD_DATASET.register_module()
class SciKnowEvalDataset(BaseDataset):

    @staticmethod
    def load(path: str, prompt_mode: str, **kwargs):

        def capitalize_first_letter(s):
            if not s:  # 检查字符串是否为空
                return s
            return s[0].upper() + s[1:]

        subset = kwargs['subset']
        data_files = {}
        test_file = f'data/{capitalize_first_letter(subset)}/'
        test_file += f'sciknoweval_{subset}_test.jsonl'
        data_files['test'] = test_file
        dataset = load_dataset(path, data_files=data_files, split='test')
        # dataset = dataset.select(range(20))
        if prompt_mode == 'zero-shot':
            dataset = dataset.map(
                lambda item: _parse(item, prompt_mode, subset),
                load_from_cache_file=False)
        elif prompt_mode == 'few-shot':
            pass  # TODO: Implement few-shot prompt
        return dataset


class SciKnowEvalEvaluator(BaseEvaluator):

    def score(self, predictions, references, test_set):
        method = test_set['prompt_mode'][0]

        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}
        correct = 0
        count = 0
        details = []
        for idx, (i, j) in enumerate(zip(predictions, references)):
            i = answer_cleansing(method, i, test_set['choices'][idx]['label'],
                                 test_set['answerKey'][idx])
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
