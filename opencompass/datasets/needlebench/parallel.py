import json
import random

import tiktoken
from datasets import Dataset
from huggingface_hub import hf_hub_download

from opencompass.datasets.base import BaseDataset
from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET


def get_unique_entries(file_path,
                       n,
                       language,
                       unique_arg1=False,
                       unique_arg2=False,
                       unique_combination=False):
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

        if (key1 not in seen_arg1 or not unique_arg1) and \
           (key2 not in seen_arg2 or not unique_arg2) and \
           (combination not in seen_combinations or not unique_combination):
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
        path: str,  # depreciated
        needle_file_name: str,
        length: int,
        depths: list[int],
        tokenizer_model: str,
        file_list: list[str],
        num_repeats_per_file: int,
        length_buffer: int,
        guide: bool,
        language: str,
        position: str = 'End',
    ):
        data = {'prompt': [], 'answer': []}
        tokenizer = tiktoken.encoding_for_model(tokenizer_model)

        repo_id = 'opencompass/NeedleBench'
        file_names = [
            'PaulGrahamEssays.jsonl', 'needles.jsonl', 'zh_finance.jsonl',
            'zh_game.jsonl', 'zh_general.jsonl', 'zh_government.jsonl',
            'zh_movie.jsonl', 'zh_tech.jsonl'
        ]

        downloaded_files = []
        for file_name in file_names:
            file_path = hf_hub_download(repo_id=repo_id,
                                        filename=file_name,
                                        repo_type='dataset')
            downloaded_files.append(file_path)

        for file in downloaded_files:
            if file.split('/')[-1] == needle_file_name:
                needle_file_path = file

        predefined_needles_bak = get_unique_entries(needle_file_path,
                                                    len(depths),
                                                    language,
                                                    unique_arg1=True,
                                                    unique_arg2=True,
                                                    unique_combination=True)

        def _generate_context(tokens_context, depths, needles):
            insertion_points = [
                int(len(tokens_context) * (depth / 100)) for depth in depths
            ]

            cumulative_inserted_length = 0

            for i, needle in enumerate(needles):
                needle_tokens = _get_tokens_from_context(needle)
                current_insertion_point = min(
                    insertion_points[i] + cumulative_inserted_length,
                    len(tokens_context))

                tokens_context = tokens_context[:current_insertion_point] + \
                    needle_tokens + tokens_context[current_insertion_point:]
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

        def _modify_retrieval_question(retrieval_question):
            if language == 'Chinese':
                parts = retrieval_question.split('请按照')
                guide_retrieval_question = (parts[0] + '在回答之前，请思考文档中与此问题'
                                            '最相关的内容是什么。请按照' + parts[1])
                return guide_retrieval_question
            elif language == 'English':
                parts = retrieval_question.split('Please answer in the format')
                guide_retrieval_question = (
                    parts[0] + 'Before answering, please consider'
                    ' what in the document is most relevant to this question.'
                    ' Please answer in the format' + parts[1])
                return guide_retrieval_question
            else:
                raise ValueError(f"Language '{language}' is not supported.")

        def _generate_prompt(context, retrieval_question):
            if guide:
                retrieval_question = _modify_retrieval_question(
                    retrieval_question)

            if language == 'Chinese':
                if position == 'End':
                    prompt = ('你是一个善于回答用户问题的智能AI助手\n'
                              '请保持你的回答简洁清楚。不要说和下面文档中的无关的话'
                              '，或重复你的回答\n请先仔细阅读下面的文档再依次回答'
                              f'最后提出的问题\n用户现在给你的文档是{context}\n\n'
                              f'现在请问：{retrieval_question}\n')
                elif position == 'Start':
                    prompt = ('你是一个善于回答用户问题的智能AI助手\n'
                              '请保持你的回答简洁清楚。不要说和下面文档中的无关的话'
                              '，或重复你的回答\n请先仔细阅读下面的文档再依次回答'
                              f'最后提出的问题\n现在请问：{retrieval_question}\n\n'
                              f'用户现在给你的文档是{context}\n')
                else:
                    raise ValueError(f'Unsupported position {position}. '
                                     'Position must be "End" or "Start".')

            elif language == 'English':
                if position == 'End':
                    prompt = (
                        'You are an intelligent AI assistant skilled in '
                        'answering user questions.\n'
                        'Please keep your answers concise and clear. Do not'
                        ' talk about irrelevant topics or repeat your '
                        'answers.\n'
                        f'The document given to you by the user is {context}'
                        f'\n\nNow, the questions are: {retrieval_question}\n')
                elif position == 'Start':
                    prompt = (
                        'You are an intelligent AI assistant skilled in '
                        'answering user questions.\n'
                        'Please keep your answers concise and clear. Do not'
                        ' talk about irrelevant topics or repeat your '
                        'answers.\n'
                        f'\nNow, the questions are: {retrieval_question}\n\n'
                        f'The document given to you by the user is {context}')
                else:
                    raise ValueError(f'Unsupported position {position}. '
                                     'Position must be "End" or "Start".')
            else:
                raise ValueError(f"Language '{language}' is not supported.")

            return prompt

        for file_path in downloaded_files:
            if file_path.split('/')[-1] not in file_list:
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
                    retrieval_question = questions + "请按照'" + \
                        answers_format + "'的格式回答。"
                elif language == 'English':
                    questions = '、'.join([
                        item['retrieval_question'].split('?')[0] + '?'
                        for item in predefined_needles
                    ])

                    answers_format = '、'.join([
                        item['retrieval_question'].split("'")[1].split('.')[0]
                        for item in predefined_needles
                    ])
                    retrieval_question = questions + \
                        "Please answer in the format of '" + \
                        answers_format + "'"

                context_length = length - length_buffer
                target_length_per_record = context_length - \
                    sum(len(tokens) for tokens
                        in _get_tokens_from_context(needles))
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

    def levenshtein_distance(self, s1, s2):
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

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
            **flattened_scores, 'details': details,
            'average_score': average_score
        }
        return result
