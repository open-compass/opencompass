# flake8: noqa: E501
import json
import os
import random
import re

import tiktoken
from datasets import Dataset

from opencompass.datasets.base import BaseDataset
from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.utils import get_data_path


def get_random_line_by_language(counter, file_path, language):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = [
            json.loads(line.strip()) for line in file
            if json.loads(line.strip())['language'] == language
        ]

    if lines:
        random.seed(counter)
        random_line = random.choice(lines)
        return {
            'needle': random_line['needle'],
            'retrieval_question': random_line['retrieval_question'],
            'keyword': random_line['arg2']
        }
    else:
        return None


@LOAD_DATASET.register_module()
class NeedleBenchOriginDataset(BaseDataset):

    @staticmethod
    def load(
        path: str,
        length: int,
        depth: int,
        tokenizer_model: str,
        file_list: list[str],
        num_repeats_per_file: int,
        length_buffer: int,
        language: str,
        needle_file_name: str,
        quesiton_position: str = 'End',
    ):
        data = {'prompt': [], 'answer': []}
        tokenizer = tiktoken.encoding_for_model(tokenizer_model)

        def _generate_context(tokens_context, depth_percent, needle):
            tokens_needle = _get_tokens_from_context(needle)
            insertion_point = int(len(tokens_context) * (depth_percent / 100))
            tokens_context = (tokens_context[:insertion_point] +
                              tokens_needle + tokens_context[insertion_point:])
            new_context = _decode_tokens(tokens_context)
            return new_context

        def _get_tokens_from_context(context):
            return tokenizer.encode(context)

        def _decode_tokens(tokens):
            return tokenizer.decode(tokens)

        def _generate_prompt(context, retrieval_question):

            if language == 'Chinese':
                if quesiton_position == 'End':
                    prompt = f'''这是一个长文本能力的测试，你需要首先阅读下面的长文档，然后根据文档中的信息回答最后的问题。
长文档的内容如下

<文档>
{context}
</文档>

根据文档中的信息，现在请问：{retrieval_question}
'''
                elif quesiton_position == 'Start':
                    prompt = f'''这是一个长文本能力的测试，你需要首先阅读下面的问题，然后根据最后长文档中的信息回答下面的问题。
现在请问：{retrieval_question}

长文档内容的如下

<文档>
{context}
</文档>

'''
                else:
                    raise ValueError('Unsupported quesiton_position. '
                                     'Position must be "End" or "Start".')
            elif language == 'English':
                if quesiton_position == 'End':
                    prompt = f'''This is a test of long-text capability. You need to first read the long document below, and then answer the final question based on the information in the document.
The content of the long document is as follows

<Document>
{context}
</Document>

Based on the information in the document, now please answer: {retrieval_question}
'''
                elif quesiton_position == 'Start':
                    prompt = f'''This is a test of long-text capability. You need to first read the question below, and then answer it based on the information in the long document that follows.
Now please answer: {retrieval_question}

The content of the long document is as follows

<Document>
{context}
</Document>

'''
                else:
                    raise ValueError(
                        f'Unsupported quesiton_position {quesiton_position}. '
                        'Position must be "End" or "Start".')
            else:
                raise ValueError(f"Language '{language}' is not supported.")

            return prompt

        file_names = [
            'en_un_asr.jsonl', 'zh_all.jsonl', 'PaulGrahamEssays.jsonl',
            'multi_needle_reasoning_en.json', 'multi_needle_reasoning_zh.json',
            'zh_finance.jsonl', 'zh_game.jsonl', 'zh_general.jsonl',
            'zh_government.jsonl', 'zh_movie.jsonl', 'zh_tech.jsonl'
        ]
        path = get_data_path(path)
        if os.environ.get('DATASET_SOURCE') == 'HF':
            from huggingface_hub import snapshot_download
            path = snapshot_download(repo_id=path, repo_type='dataset')
        needle_file_path = os.path.join(path, needle_file_name)

        for file_name in file_names:
            file_path = os.path.join(path, file_name)
            if file_name not in file_list:
                continue

            with open(file_path, 'r', encoding='utf-8') as f:
                lines_bak = [json.loads(line.strip()) for line in f]
            lines = lines_bak.copy()
            for counter in range(num_repeats_per_file):
                random.seed(counter)
                random.shuffle(lines)
                random_needle = get_random_line_by_language(
                    counter, needle_file_path, language)
                needle = '\n' + random_needle['needle'] + '\n'
                retrieval_question = random_needle['retrieval_question']
                keyword = random_needle['keyword']

                context_length = length - length_buffer
                target_length_per_record = context_length - len(
                    _get_tokens_from_context(needle))
                target_length_per_record = max(target_length_per_record, 0)
                accumulated_tokens = []
                for line in lines:
                    tokens_current_line = _get_tokens_from_context(
                        line['text'])
                    accumulated_tokens.extend(tokens_current_line)

                    if len(accumulated_tokens) >= target_length_per_record:
                        break

                processed_text = _generate_context(
                    accumulated_tokens[:target_length_per_record], depth,
                    needle)

                processed_prompt = _generate_prompt(processed_text,
                                                    retrieval_question)

                data['prompt'].append(processed_prompt)
                data['answer'].append(needle + '*' + keyword)

        dataset = Dataset.from_dict({
            'prompt': data['prompt'],
            'answer': data['answer'],
        })
        return dataset


class NeedleBenchOriginEvaluator(BaseEvaluator):

    def score(self, predictions, gold):

        if len(predictions) != len(gold):
            return {'error': 'predictions and gold have different lengths'}

        total_score = 0
        details = []
        for prediction, reference in zip(predictions, gold):
            keyword = reference.split('*')[1]
            reference = reference.split('*')[0]
            raw_prediction = prediction
            prediction = re.sub(r'\s+', '', prediction)
            reference = re.sub(r'\s+', '', reference)

            if keyword in raw_prediction:
                score = 100
            else:
                score = 0

            detail = {'pred': prediction, 'answer': reference, 'score': score}
            total_score += score
            details.append(detail)

        average_score = total_score / len(predictions) if predictions else 0
        result = {'score': average_score, 'details': details}
        return result


@TEXT_POSTPROCESSORS.register_module('needlebench')
def needlebench_postprocess(text: str) -> str:
    return text


@TEXT_POSTPROCESSORS.register_module('needlebench_dataset_postprocess')
def needlebench_dataset_postprocess(text: str) -> str:
    return text
