import json
import os
import random

import tiktoken
from datasets import Dataset
from huggingface_hub import hf_hub_download

from opencompass.datasets.base import BaseDataset
from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET


def get_random_needles(counter, file_path, needle_count):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    matching_records = [
        record for record in data
        if record.get('derivation_count') == needle_count
    ]

    if matching_records:
        random.seed(counter)
        random_record = random.choice(matching_records)
        return {
            'needles': random_record['derivations'],
            'answer': random_record['answer'],
            'retrieval_question': random_record['question']
        }
    else:
        return None


@LOAD_DATASET.register_module()
class NeedleBenchMultiDataset(BaseDataset):

    @staticmethod
    def load(
        path: str,  # depreciated
        length: int,
        depth: int,
        tokenizer_model: str,
        file_list: 'list[str]',
        num_repeats_per_file: int,
        length_buffer: int,
        guide: bool,
        language: str,
        needle_file_name: str,
        num_needles: int,
        diff: int,
        position: str = 'End',
    ):
        data = {'prompt': [], 'answer': []}
        tokenizer = tiktoken.encoding_for_model(tokenizer_model)

        def _generate_context(tokens_context, depth_percent, needles):
            tokens_needle = [
                _get_tokens_from_context(needle) for needle in needles
            ]
            insertion_points = []
            total_length = len(tokens_context)

            for i, needle_tokens in enumerate(tokens_needle):
                if i == 0:
                    insertion_point = int(total_length * (depth_percent / 100))
                else:
                    insertion_point = int(insertion_points[i - 1] +
                                          len(tokens_needle[i - 1]) +
                                          total_length * (diff / 100))
                insertion_point = min(
                    insertion_point,
                    total_length + sum(len(tn) for tn in tokens_needle[:i]))
                insertion_points.append(insertion_point)

            for i, needle_tokens in enumerate(tokens_needle):
                tokens_context = tokens_context[:insertion_points[i]] \
                    + needle_tokens + tokens_context[insertion_points[i]:]
                for j in range(i + 1, len(insertion_points)):
                    insertion_points[j] += len(needle_tokens)

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
                guide_retrieval_question = (retrieval_question +
                                            '在回答之前，请思考文档中与此问题'
                                            '最相关的内容是什么。')
                return guide_retrieval_question
            elif language == 'English':
                guide_retrieval_question = (
                    retrieval_question + 'Before answering, please consider'
                    ' what in the document is most relevant to this question.')
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
                              '，或重复你的回答\n'
                              f'用户现在给你的文档是{context}\n\n'
                              f'现在请问：{retrieval_question}')
                elif position == 'Start':
                    prompt = ('你是一个善于回答用户问题的智能AI助手\n'
                              '请保持你的回答简洁清楚。不要说和下面文档中的无关的话'
                              '，或重复你的回答\n'
                              f'现在请问：{retrieval_question}',
                              f'用户现在给你的文档是{context}\n\n')
                else:
                    raise ValueError('Unsupported position. '
                                     'Position must be "End" or "Start".')
            elif language == 'English':
                if position == 'End':
                    prompt = ('You are an intelligent AI assistant skilled in '
                              'answering user questions.\n'
                              'Please keep your answers concise and clear. Do '
                              'not talk about irrelevant topics or repeat '
                              'your answers.\nThe document '
                              f'given to you by the user is {context}\n\n'
                              f'Now, the question is: {retrieval_question}')
                elif position == 'Start':
                    prompt = ('You are an intelligent AI assistant skilled in '
                              'answering user questions.\n'
                              'Please keep your answers concise and clear. Do '
                              'not talk about irrelevant topics or repeat '
                              'your answers.\n'
                              f'Now, the question is: {retrieval_question}'
                              'The document given to you by the user'
                              f' is {context}\n\n')
                else:
                    raise ValueError(f'Unsupported position {position}. '
                                     'Position must be "End" or "Start".')
            else:
                raise ValueError(f"Language '{language}' is not supported.")

            return prompt

        repo_id = 'opencompass/NeedleBench'
        file_names = [
            'PaulGrahamEssays.jsonl', 'multi_needle_reasoning_en.json',
            'multi_needle_reasoning_zh.json', 'zh_finance.jsonl',
            'zh_game.jsonl', 'zh_general.jsonl', 'zh_government.jsonl',
            'zh_movie.jsonl', 'zh_tech.jsonl'
        ]
        downloaded_files = []
        base_file_path = ''
        for file_name in file_names:
            file_path = hf_hub_download(repo_id=repo_id,
                                        filename=file_name,
                                        repo_type='dataset')
            downloaded_files.append(file_path)
            base_file_path = '/'.join(file_path.split('/')[:-1])

        needle_file_path = os.path.join(base_file_path, needle_file_name)
        for file_path in downloaded_files:
            if file_path.split('/')[-1] not in file_list:
                continue

            with open(file_path, 'r', encoding='utf-8') as f:
                lines_bak = [json.loads(line.strip()) for line in f]
            lines = lines_bak.copy()
            for counter in range(num_repeats_per_file):
                random.seed(counter)
                random.shuffle(lines)
                random_needle_data = get_random_needles(
                    counter, needle_file_path, num_needles)
                needles = [
                    '\n' + needle + '\n'
                    for needle in random_needle_data['needles']
                ]
                answer = random_needle_data['answer']
                keyword = answer
                retrieval_question = random_needle_data['retrieval_question']
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
                    accumulated_tokens[:target_length_per_record], depth,
                    needles)

                processed_prompt = _generate_prompt(processed_text,
                                                    retrieval_question)

                data['prompt'].append(processed_prompt)
                data['answer'].append(answer + '*' + keyword)

        dataset = Dataset.from_dict({
            'prompt': data['prompt'],
            'answer': data['answer'],
        })
        return dataset


class NeedleBenchMultiEvaluator(BaseEvaluator):

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

        total_score = 0
        details = []

        for prediction, reference in zip(predictions, gold):
            answer, keyword = reference.split('*')
            keywords = keyword.lower().split()
            prediction = prediction.lower()

            keyword_score = 100 / len(keywords) if keywords else 0

            matched_keywords = sum(1 for kword in keywords
                                   if kword in prediction)
            score = matched_keywords * keyword_score

            detail = {
                'pred': prediction,
                'answer': reference,
                'matched_keywords': matched_keywords,
                'score': score
            }

            total_score += score
            details.append(detail)

        average_score = total_score / len(predictions) if predictions else 0
        return {'score': average_score, 'details': details}
