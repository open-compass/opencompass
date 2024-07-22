import json
import os
import random
import re

import tiktoken
from datasets import Dataset
from huggingface_hub import hf_hub_download

from opencompass.datasets.base import BaseDataset
from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS


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
        path: str,  # depreciated
        length: int,
        depth: int,
        tokenizer_model: str,
        file_list: list[str],
        num_repeats_per_file: int,
        length_buffer: int,
        guide: bool,
        language: str,
        needle_file_name: str,
        position: str = 'End',
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
            'PaulGrahamEssays.jsonl', 'needles.jsonl', 'zh_finance.jsonl',
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

        for file_path in downloaded_files:
            if file_path.split('/')[-1] not in file_list:
                continue
            with open(file_path, 'r', encoding='utf-8') as f:
                lines_bak = [json.loads(line.strip()) for line in f]
            lines = lines_bak.copy()
            for counter in range(num_repeats_per_file):
                random.seed(counter)
                random.shuffle(lines)
                needle_file_path = os.path.join(base_file_path,
                                                needle_file_name)
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

    def __init__(self, use_trim=False):
        self.use_trim = use_trim

    @staticmethod
    def _trim_prediction(prediction, reference):
        """Trims the prediction string based on the length of the reference
        string.

        Args:
            prediction (str): The prediction string.
            reference (str): The reference string.

        Returns:
            str: The trimmed prediction string.
        """
        l08 = int(0.8 * len(reference))
        l12 = int(1.2 * len(reference))
        trimmed_prediction = prediction[:l12]

        if len(trimmed_prediction) > l08 and \
                reference[-1] in trimmed_prediction[l08:]:
            end_pos = l08 + trimmed_prediction[l08:].index(reference[-1]) + 1
            trimmed_prediction = trimmed_prediction[:end_pos]

        return trimmed_prediction

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
            keyword = reference.split('*')[1]
            reference = reference.split('*')[0]
            raw_prediction = prediction
            prediction = re.sub(r'\s+', '', prediction)
            reference = re.sub(r'\s+', '', reference)

            if self.use_trim:
                prediction = NeedleBenchOriginEvaluator._trim_prediction(
                    prediction, reference)

            edit_distance = self.levenshtein_distance(prediction, reference)
            max_len = max(len(prediction), len(reference))
            score = 100 * (1 -
                           edit_distance / max_len) if max_len != 0 else 100

            if keyword in raw_prediction:
                print(f'{keyword} is in {prediction}')
                score = 100
            else:
                print(f'{keyword} is not in {prediction}')
                score = 0.2 * score

            detail = {
                'pred': prediction,
                'answer': reference,
                'edit_distance': edit_distance,
                'score': score
            }
            total_score += score
            details.append(detail)

        average_score = total_score / len(predictions) if predictions else 0
        result = {'score': average_score, 'details': details}
        return result


@TEXT_POSTPROCESSORS.register_module('needlebench')
def needlebench_postprocess(text: str) -> str:
    return text


@TEXT_POSTPROCESSORS.register_module('needlebench_dataset')
def needlebench_dataset_postprocess(text: str) -> str:
    return text
