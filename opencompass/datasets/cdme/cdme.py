import json
import re
from pathlib import Path

import numpy as np
import tiktoken
from datasets import Dataset

from opencompass.datasets.base import BaseDataset
from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS


class CDMEDatasetProcessor:

    def __init__(self,
                 path,
                 output_path,
                 tokenizer_model='gpt-4',
                 num_records_per_file=10,
                 length_buffer=200,
                 guided=False):
        self.path = path
        self.output_path = output_path
        self.tokenizer = tiktoken.encoding_for_model(tokenizer_model)
        self.num_records_per_file = num_records_per_file
        self.length_buffer = length_buffer
        self.guided = guided

    def process_files(self,
                      context_lengths,
                      needle,
                      retrieval_question,
                      document_depth_percent_intervals,
                      document_depth_percent_interval_type='linear'):
        files = Path(self.path).glob('*.jsonl')
        for file in files:
            self.process_file(file, context_lengths, needle,
                              retrieval_question,
                              document_depth_percent_intervals,
                              document_depth_percent_interval_type)

    def process_file(self, file, context_lengths, needle, retrieval_question,
                     document_depth_percent_intervals,
                     document_depth_percent_interval_type):
        with open(file, 'r', encoding='utf-8') as f:
            lines = [json.loads(line.strip()) for line in f]

        for original_context_length in context_lengths:
            context_length = original_context_length - self.length_buffer
            target_length_per_record = context_length - len(
                self._get_tokens_from_context(needle))
            for depth_percent in self._generate_depth_percents(
                    document_depth_percent_intervals,
                    document_depth_percent_interval_type):
                output_file = (Path(self.output_path) /
                               f'Length{original_context_length}'
                               f'Depth{int(depth_percent)}' /
                               f'{file.stem}_Length{original_context_length}'
                               f'_Depth{int(depth_percent)}{file.suffix}')

                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as out_f:
                    counter = 0
                    accumulated_tokens = []
                    for line in lines:
                        tokens_current_line = self._get_tokens_from_context(
                            line['text'])
                        accumulated_tokens.extend(tokens_current_line)

                        if len(accumulated_tokens) >= target_length_per_record:

                            processed_text = self._generate_context(
                                accumulated_tokens[:target_length_per_record],
                                depth_percent, needle)

                            processed_prompt = self._generate_prompt(
                                processed_text, retrieval_question)
                            json.dump(
                                {
                                    'prompt': processed_prompt,
                                    'answer': needle
                                },
                                out_f,
                                ensure_ascii=False)
                            out_f.write('\n')
                            counter += 1
                            if counter >= self.num_records_per_file:
                                break
                            # Reset the accumulated tokens for the next record
                            accumulated_tokens = []

    def _generate_context(self, tokens_context, depth_percent, needle):
        tokens_needle = self._get_tokens_from_context(needle)

        # Insert the needle into the context at the specified depth percent
        insertion_point = int(len(tokens_context) * (depth_percent / 100))
        tokens_context = (tokens_context[:insertion_point] + tokens_needle +
                          tokens_context[insertion_point:])

        # Decode the tokens back to text
        new_context = self._decode_tokens(tokens_context)
        return new_context

    def _get_tokens_from_context(self, context):
        return self.tokenizer.encode(context)

    def _decode_tokens(self, tokens):
        return self.tokenizer.decode(tokens)

    def _generate_prompt(self, context, retrieval_question):
        if self.guided:
            prompt = ('你是一个善于回答用户问题的智能AI助手\n'
                      '请保持你的回答简洁清楚。不要说和下面文档中的无关的话，或重复你的回答\n'
                      f'用户现在给你的文档是{context}\n\n'
                      f'现在请问：{retrieval_question}'
                      f'提示：文档中与该问题最相关的句子是_______')
        else:
            prompt = ('你是一个善于回答用户问题的智能AI助手\n'
                      '请保持你的回答简洁清楚。不要说和下面文档中的无关的话，或重复你的回答\n'
                      f'用户现在给你的文档是{context}\n\n'
                      f'现在请问：{retrieval_question}')
        return prompt

    def _generate_depth_percents(self, intervals, interval_type):
        if interval_type == 'linear':
            return np.linspace(0, 100, num=intervals)
        elif interval_type == 'sigmoid':
            return [self._logistic(x) for x in np.linspace(0, 100, intervals)]
        else:
            raise ValueError('Unsupported interval type')

    @staticmethod
    def _logistic(x, L=100, x0=50, k=0.1):
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)


@LOAD_DATASET.register_module()
class CDMEDataset(BaseDataset):

    @staticmethod
    def load(path: str):

        data = {'prompt': [], 'answer': []}
        for file in Path(path).glob('*.jsonl'):
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = json.loads(line.strip())
                    data['prompt'].append(line['prompt'])
                    data['answer'].append(line['answer'])

        dataset = Dataset.from_dict({
            'prompt': data['prompt'],
            'answer': data['answer'],
        })
        return dataset


class CDMEEvaluator(BaseEvaluator):

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

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different lengths'
            }

        total_score = 0
        details = []
        for prediction, reference in zip(predictions, references):
            prediction = re.sub(r'\s+', '', prediction)
            reference = re.sub(r'\s+', '', reference)
            edit_distance = self.levenshtein_distance(prediction, reference)
            max_len = max(len(prediction), len(reference))
            score = 100 * (1 -
                           edit_distance / max_len) if max_len != 0 else 100

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


@TEXT_POSTPROCESSORS.register_module('cdme')
def cdme_postprocess(text: str) -> str:
    return text


@TEXT_POSTPROCESSORS.register_module('cdme_dataset')
def cdme_dataset_postprocess(text: str) -> str:
    return text
