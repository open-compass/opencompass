# flake8: noqa: E501
import json
import os
import random

import tiktoken
from datasets import Dataset

from opencompass.datasets.base import BaseDataset
from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path


def get_unique_entries(
    file_path,
    n,
    language,
    unique_arg1=False,
    unique_arg2=False,
    unique_combination=False,
):
    seen_arg1 = set()
    seen_arg2 = set()
    seen_combinations = set()
    results = []

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    random.shuffle(lines)

    for line in lines:
        try:
            entry = json.loads(line.strip())
        except json.JSONDecodeError:
            continue

        if entry.get('language') != language:
            continue

        key1 = entry.get('arg1', '') if unique_arg1 else ''
        key2 = entry.get('arg2', '') if unique_arg2 else ''
        combination = (key1, key2) if unique_combination else ''

        if ((key1 not in seen_arg1 or not unique_arg1)  # noqa: E501
                and (key2 not in seen_arg2 or not unique_arg2)
                and  # noqa: E501
            (combination not in seen_combinations
             or not unique_combination)):  # noqa: E501
            seen_arg1.add(key1)
            seen_arg2.add(key2)
            seen_combinations.add(combination)
            results.append(entry)

        if len(results) == n:
            break

    return results


@LOAD_DATASET.register_module()
class NeedleBenchParallelDataset(BaseDataset):

    @staticmethod
    def load(
        path: str,
        needle_file_name: str,
        length: int,
        depths: list[int],
        tokenizer_model: str,
        file_list: list[str],
        num_repeats_per_file: int,
        length_buffer: int,
        language: str,
        quesiton_position: str = 'End',
    ):
        data = {'prompt': [], 'answer': []}
        tokenizer = tiktoken.encoding_for_model(tokenizer_model)

        file_names = [
            'PaulGrahamEssays.jsonl',
            'multi_needle_reasoning_en.json',
            'multi_needle_reasoning_zh.json',
            'zh_finance.jsonl',
            'zh_game.jsonl',
            'zh_general.jsonl',
            'zh_government.jsonl',
            'zh_movie.jsonl',
            'zh_tech.jsonl',
        ]
        path = get_data_path(path)
        if os.environ.get('DATASET_SOURCE') == 'HF':
            from huggingface_hub import snapshot_download

            path = snapshot_download(repo_id=path, repo_type='dataset')
        needle_file_path = os.path.join(path, needle_file_name)

        predefined_needles_bak = get_unique_entries(
            needle_file_path,
            len(depths),
            language,
            unique_arg1=True,
            unique_arg2=True,
            unique_combination=True,
        )

        def _generate_context(tokens_context, depths, needles):
            insertion_points = [
                int(len(tokens_context) * (depth / 100)) for depth in depths
            ]

            cumulative_inserted_length = 0

            for i, needle in enumerate(needles):
                needle_tokens = _get_tokens_from_context(needle)
                current_insertion_point = min(
                    insertion_points[i] + cumulative_inserted_length,
                    len(tokens_context),
                )

                tokens_context = (tokens_context[:current_insertion_point] +
                                  needle_tokens +
                                  tokens_context[current_insertion_point:])
                cumulative_inserted_length += len(needle_tokens)

            new_context = _decode_tokens(tokens_context)
            return new_context

        def _get_tokens_from_context(context):
            if isinstance(context, list):
                return [tokenizer.encode(item) for item in context]
            else:
                return tokenizer.encode(context)

        def _decode_tokens(tokens):
            return tokenizer.decode(tokens)

        def _generate_prompt(context, retrieval_question):
            if language == 'Chinese':
                if quesiton_position == 'End':
                    prompt = f'''这是一个长文本能力的测试，你需要首先阅读下面的长文档，然后根据文档中的信息，依次回答最后的问题。
长文档的内容如下

<文档>
{context}
</文档>

根据文档中的信息，现在请问：{retrieval_question}
'''
                elif quesiton_position == 'Start':
                    prompt = f'''这是一个长文本能力的测试，你需要首先阅读下面的问题，然后根据最后长文档中的信息，依次回答下面的问题。
现在请问：{retrieval_question}

长文档内容的如下

<文档>
{context}
</文档>

'''
                else:
                    raise ValueError(
                        f'Unsupported quesiton_position {quesiton_position}. '
                        'Position must be "End" or "Start".')
            elif language == 'English':
                if quesiton_position == 'End':
                    prompt = f'''This is a test of long-text capability. You need to first read the long document below, and then answer the final questions one by one based on the information in the document.
The content of the long document is as follows

<Document>
{context}
</Document>

Based on the information in the document, now please answer: {retrieval_question}
'''
                elif quesiton_position == 'Start':
                    prompt = f'''This is a test of long-text capability. You need to first read the questions below, and then answer them one by one based on the information in the long document that follows.
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
                predefined_needles = predefined_needles_bak.copy()
                random.seed(counter)
                random.shuffle(predefined_needles)

                needles = [
                    '\n' + item['needle'] + '\n' for item in predefined_needles
                ]
                keywords = [item['arg2'] for item in predefined_needles]
                if language == 'Chinese':
                    questions = '、'.join([
                        item['retrieval_question'].split('？')[0] + '？'
                        for item in predefined_needles
                    ])

                    answers_format = '、'.join([
                        item['retrieval_question'].split("'")[1].split('。')[0]
                        for item in predefined_needles
                    ])
                    retrieval_question = (questions + "请按照'" + answers_format +
                                          "'的格式回答。")
                elif language == 'English':
                    questions = '、'.join([
                        item['retrieval_question'].split('?')[0] + '?'
                        for item in predefined_needles
                    ])

                    answers_format = '、'.join([
                        item['retrieval_question'].split("'")[1].split('.')[0]
                        for item in predefined_needles
                    ])
                    retrieval_question = (questions +
                                          "Please answer in the format of '" +
                                          answers_format + "'")

                context_length = length - length_buffer
                target_length_per_record = context_length - sum(
                    len(tokens)
                    for tokens in _get_tokens_from_context(needles))
                target_length_per_record = max(target_length_per_record, 0)
                accumulated_tokens = []
                for line in lines:
                    tokens_current_line = _get_tokens_from_context(
                        line['text'])
                    accumulated_tokens.extend(tokens_current_line)

                    if len(accumulated_tokens) >= target_length_per_record:
                        break

                processed_text = _generate_context(
                    accumulated_tokens[:target_length_per_record], depths,
                    needles)

                processed_prompt = _generate_prompt(processed_text,
                                                    retrieval_question)

                data['prompt'].append(processed_prompt)

                data['answer'].append('*'.join(keywords) + '#' +
                                      '*'.join(map(str, depths)))

        dataset = Dataset.from_dict({
            'prompt': data['prompt'],
            'answer': data['answer'],
        })
        return dataset


class NeedleBenchParallelEvaluator(BaseEvaluator):

    def score(self, predictions, gold):
        if len(predictions) != len(gold):
            return {'error': 'predictions and gold have different lengths'}
        print('predictions:', predictions)
        print('gold:', gold)

        details = []
        depths = [int(i) for i in gold[0].split('#')[1].split('*')]
        scores_by_depth = {depth: 0 for depth in depths}

        for prediction, reference in zip(predictions, gold):
            print(reference)
            keywords = reference.split('#')[0].split('*')
            print(keywords)
            for keyword, depth in zip(keywords, depths):
                print('iterating:', keyword, depth)
                if keyword in prediction:
                    print(f'{keyword} at depth {depth} is in {prediction}')
                    scores_by_depth[depth] += 100 / (len(predictions))

        average_score = sum(scores_by_depth.values()) / len(scores_by_depth)

        flattened_scores = {
            'Depth' + str(depth): score
            for depth, score in scores_by_depth.items()
        }

        result = {
            **flattened_scores,
            'details': details,
            'average_score': average_score,
        }
        return result
