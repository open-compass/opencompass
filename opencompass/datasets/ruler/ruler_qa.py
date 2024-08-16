# flake8: noqa: F401, E501
import json
import os
import random

import numpy as np
import requests
import tiktoken
from datasets import Dataset
from transformers import AutoTokenizer

from opencompass.datasets.base import BaseDataset
from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path


@LOAD_DATASET.register_module()
class RulerQaDataset(BaseDataset):

    @staticmethod
    def load(
        path: str,
        dataset: str = 'squad',
        max_seq_length: int = 4096,
        tokenizer_model: str = 'gpt-4',
        template:
        str = 'Answer the question based on the given documents. Only give me the answer and do not output any other words.\n\nThe following are given documents.\n\n{context}\n\nAnswer the question based on the given documents. Only give me the answer and do not output any other words.\n\nQuestion: {query} Answer:',
        tokens_to_generate: int = 32,
        num_samples: int = 500,
        pre_samples: int = 0,
        random_seed: int = 42,
        remove_newline_tab: str = '',
    ) -> Dataset:

        if tokenizer_model == 'gpt-4':
            tokenizer = tiktoken.encoding_for_model(tokenizer_model)
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_model,
                                                      trust_remote_code=True)

        random.seed(random_seed)
        np.random.seed(random_seed)

        # Read SQuAD QA dataset
        def _read_squad(file):
            file = get_data_path(file, local_mode=True)
            with open(file) as f:
                data = json.load(f)

            total_docs = [
                p['context'] for d in data['data'] for p in d['paragraphs']
            ]
            total_docs = sorted(list(set(total_docs)))
            total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}

            total_qas = []
            for d in data['data']:
                more_docs = [
                    total_docs_dict[p['context']] for p in d['paragraphs']
                ]
                for p in d['paragraphs']:
                    for qas in p['qas']:
                        if not qas['is_impossible']:
                            total_qas.append({
                                'query':
                                qas['question'],
                                'outputs': [a['text'] for a in qas['answers']],
                                'context': [total_docs_dict[p['context']]],
                                'more_context': [
                                    idx for idx in more_docs
                                    if idx != total_docs_dict[p['context']]
                                ],
                            })

            return total_qas, total_docs

        # Read Hotpot QA dataset
        def _read_hotpotqa(file_path):
            # url = (
            #     'http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json'
            # )
            # if not os.path.exists(os.path.dirname(file_path)):
            #     os.makedirs(os.path.dirname(file_path))

            # if not os.path.exists(file_path):
            #     response = requests.get(url)
            #     if response.status_code == 200:
            #         with open(file_path, 'wb') as file:
            #             file.write(response.content)
            #     else:
            #         print(
            #             f'Failed to download file. Status code: {response.status_code}'
            #         )
            # else:
            #     print('File already exists.')
            file_path = get_data_path(file_path, local_mode=True)

            with open(file_path) as f:
                data = json.load(f)

            total_docs = [
                f"{t}\n{''.join(p)}" for d in data for t, p in d['context']
            ]
            total_docs = sorted(list(set(total_docs)))
            total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}

            total_qas = []
            for d in data:
                total_qas.append({
                    'query':
                    d['question'],
                    'outputs': [d['answer']],
                    'context': [
                        total_docs_dict[f"{t}\n{''.join(p)}"]
                        for t, p in d['context']
                    ],
                })

            return total_qas, total_docs

        DOCUMENT_PROMPT = 'Document {i}:\n{document}'
        if dataset == 'squad':
            QAS, DOCS = _read_squad(path)
        elif dataset == 'hotpotqa':
            QAS, DOCS = _read_hotpotqa(path)
        else:
            raise NotImplementedError(f'{dataset} is not implemented.')

        def generate_input_output(index, num_docs):
            curr_q = QAS[index]['query']
            curr_a = QAS[index]['outputs']
            curr_docs = QAS[index]['context']
            curr_more = QAS[index].get('more_context', [])
            if num_docs < len(DOCS):
                if (num_docs - len(curr_docs)) > len(curr_more):
                    addition_docs = [
                        i for i, d in enumerate(DOCS)
                        if i not in curr_docs + curr_more
                    ]
                    all_docs = (curr_docs + curr_more + random.sample(
                        addition_docs,
                        max(0, num_docs - len(curr_docs) - len(curr_more)),
                    ))
                else:
                    all_docs = curr_docs + random.sample(
                        curr_more, num_docs - len(curr_docs))

                all_docs = [DOCS[idx] for idx in all_docs]
            else:
                all_docs = DOCS

            random.Random(random_seed).shuffle(all_docs)

            context = '\n\n'.join([
                DOCUMENT_PROMPT.format(i=i + 1, document=d)
                for i, d in enumerate(all_docs)
            ])
            input_text = template.format(context=context, query=curr_q)
            return input_text, curr_a

        def _generate_samples(num_samples: int,
                              max_seq_length: int,
                              incremental: int = 10):

            data = {'prompt': [], 'answer': []}

            # Find the perfect num_docs
            num_docs = incremental

            total_tokens = 0  # Track the total tokens generated for this example
            while total_tokens + tokens_to_generate < max_seq_length:
                input_text, answer = generate_input_output(0, num_docs)
                # Calculate the number of tokens in the example
                total_tokens = len(tokenizer.encode(input_text + f' {answer}'))
                print(
                    f'Max length {max_seq_length} | Current length {total_tokens + tokens_to_generate} | Docs: {num_docs}'
                )
                if total_tokens + tokens_to_generate > max_seq_length:
                    num_docs -= incremental
                    break

                num_docs += incremental
                if num_docs > len(DOCS):
                    num_docs = len(DOCS)
                    break
            print('Number of documents:', num_docs)

            # Generate samples
            for index in range(num_samples):
                used_docs = num_docs
                while True:
                    try:
                        input_text, answer = generate_input_output(
                            index + pre_samples, used_docs)
                        length = len(
                            tokenizer.encode(input_text)) + tokens_to_generate
                        assert (length <= max_seq_length
                                ), f'{length} exceeds max_seq_length.'
                        break
                    except:
                        if used_docs > incremental:
                            used_docs -= incremental

                if remove_newline_tab:
                    input_text = ' '.join(
                        input_text.replace('\n',
                                           ' ').replace('\t',
                                                        ' ').strip().split())

                data['prompt'].append(input_text)
                data['answer'].append(answer)

            return data

        # Generate Data
        data = _generate_samples(num_samples=num_samples,
                                 max_seq_length=max_seq_length)
        dataset = Dataset.from_dict(data)
        return dataset


class RulerQaEvaluator(BaseEvaluator):

    def score(self, predictions, gold):
        score = (sum([
            max([1.0 if r.lower() in pred.lower() else 0.0 for r in ref])
            for pred, ref in zip(predictions, gold)
        ]) / len(predictions) * 100)
        result = {'score': round(score, 2)}
        return result
