# flake8: noqa: F401, E501
import json
import os
import random
import re
import uuid
from pathlib import Path

import numpy as np
import tiktoken
from datasets import Dataset
from transformers import AutoTokenizer

from opencompass.datasets.base import BaseDataset
from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path


@LOAD_DATASET.register_module()
class RulerNiahDataset(BaseDataset):

    @staticmethod
    def load(
        base_path: str,
        file_path: str,
        tokens_to_generate: int = 128,
        max_seq_length: int = 4096,
        tokenizer_model: str = 'gpt-4',
        num_samples: int = 500,
        random_seed: int = 42,
        template:
        str = 'Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat are all the special magic {type_needle_v} for {query} mentioned in the provided text? The special magic {type_needle_v} for {query} mentioned in the provided text are',
        num_needle_k: int = 1,
        num_needle_v: int = 1,
        num_needle_q: int = 1,
        type_haystack: str = 'essay',
        type_needle_k: str = 'words',
        type_needle_v: str = 'numbers',
        remove_newline_tab: str = '',
    ) -> Dataset:

        data = {'prompt': [], 'answer': []}
        if tokenizer_model == 'gpt-4':
            tokenizer = tiktoken.encoding_for_model(tokenizer_model)
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_model,
                                                      trust_remote_code=True)

        random.seed(random_seed)
        np.random.seed(random_seed)
        num_needle_k = max(num_needle_k, num_needle_q)

        # Define Needle/Haystack Format
        needle = 'One of the special magic {type_needle_v} for {key} is: {value}.'
        if type_haystack == 'essay':
            essay = os.path.join(base_path, file_path)
            essay = get_data_path(essay, local_mode=True)
            # essay = json.load(open(essay))['text']
            combined_essay = ''
            with open(essay, 'r', encoding='utf-8') as f:
                for line in f:
                    line_text = json.loads(line.strip()).get('text',
                                                             '').strip()
                    combined_essay += line_text + ' '
            haystack = re.sub(r'\s+', ' ', combined_essay).split(' ')
        elif type_haystack == 'repeat':
            haystack = 'The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.'
        elif type_haystack == 'needle':
            haystack = needle
        else:
            raise NotImplementedError(f'{type_haystack} is not implemented.')
        try:
            import wonderwords
        except ImportError:
            raise ImportError('''Please install wonderwords by:
                              pip install wonderwords''')
        # Words
        nouns = wonderwords.random_word._get_words_from_text_file(
            'nounlist.txt')
        adjs = wonderwords.random_word._get_words_from_text_file(
            'adjectivelist.txt')
        # verbs = wonderwords.random_word._get_words_from_text_file("verblist.txt")
        words = [f'{adj}-{noun}' for adj in adjs for noun in nouns]
        words = sorted(list(set(words)))

        # Positions
        DEPTHS = list(
            np.round(np.linspace(0, 100, num=40, endpoint=True)).astype(int))

        def _generate_random_number(num_digits=7):
            lower_bound = 10**(num_digits - 1)
            upper_bound = 10**num_digits - 1
            return str(random.randint(lower_bound, upper_bound))

        def _generate_random_word():
            word = random.choice(words)
            return word

        def _generate_random_uuid():
            return str(uuid.UUID(int=random.getrandbits(128), version=4))

        def _generate_random(type_needle: str):
            if type_needle == 'numbers':
                return _generate_random_number()
            elif type_needle == 'words':
                return _generate_random_word()
            elif type_needle == 'uuids':
                return _generate_random_uuid()
            else:
                raise NotImplementedError(f'{type_needle} is not implemented.')

        def _generate_input_output(num_haystack, type_needle_v, template):
            keys, values, needles = [], [], []
            for _ in range(num_needle_k):
                keys.append(_generate_random(type_needle_k))
                value = []
                for _ in range(num_needle_v):
                    value.append(_generate_random(type_needle_v))
                    needles.append(
                        needle.format(
                            type_needle_v=type_needle_v,
                            key=keys[-1],
                            value=value[-1],
                        ))
                values.append(value)

            random.Random(random_seed).shuffle(needles)

            # Context
            if type_haystack == 'essay':
                text = ' '.join(haystack[:num_haystack])
                # document_sents = sent_tokenize(text.strip())
                document_sents = text.split('. ')
                document_sents = [
                    sentence.strip() for sentence in document_sents if sentence
                ]  # remove possible whitespace
                insertion_positions = ([0] + sorted([
                    int(len(document_sents) * (depth / 100))
                    for depth in random.sample(DEPTHS, len(needles))
                ]) + [len(document_sents)])
                document_sents_list = []
                for i in range(1, len(insertion_positions)):
                    last_pos = insertion_positions[i - 1]
                    next_pos = insertion_positions[i]
                    document_sents_list.append(' '.join(
                        document_sents[last_pos:next_pos]))
                    if i - 1 < len(needles):
                        document_sents_list.append(needles[i - 1])
                context = ' '.join(document_sents_list)

            else:
                if type_haystack == 'repeat':
                    sentences = [haystack] * num_haystack
                elif type_haystack == 'needle':
                    sentences = [
                        haystack.format(
                            type_needle_v=type_needle_v,
                            key=_generate_random(type_needle_k),
                            value=_generate_random(type_needle_v),
                        ) for _ in range(num_haystack)
                    ]

                indexes = sorted(random.sample(range(num_haystack),
                                               len(needles)),
                                 reverse=True)
                for index, element in zip(indexes, needles):
                    sentences.insert(index, element)
                context = '\n'.join(sentences)

            ## Query and Answer
            indices = random.sample(range(num_needle_k), num_needle_q)
            queries = [keys[i] for i in indices]
            answers = [a for i in indices for a in values[i]]
            query = (', '.join(queries[:-1]) + ', and ' +
                     queries[-1] if len(queries) > 1 else queries[0])

            if num_needle_q * num_needle_v == 1:
                template = template.replace('Some', 'A')
                template = template.replace('are all', 'is')
                template = template.replace('are', 'is')
                template = template.replace('answers', 'answer')
                type_needle_v = type_needle_v[:-1]  # remove "s"

            input_text = template.format(
                type_needle_v=type_needle_v,
                context=context,
                query=query,
            )

            return input_text, answers

        # Generate Samples
        if type_haystack == 'essay':
            incremental = 500
        elif type_haystack == 'repeat':
            incremental = 25
        elif type_haystack == 'needle':
            incremental = 25

        if type_haystack != 'essay' and max_seq_length < 4096:
            incremental = 5

        num_haystack = incremental
        print('Num haystack:', num_haystack)

        total_tokens = 0  # Track the total tokens generated for the first example
        while total_tokens + tokens_to_generate < max_seq_length:
            input_text, answer = _generate_input_output(
                num_haystack, type_needle_v, template)
            # Calculate the number of tokens in the example
            total_tokens = len(tokenizer.encode(input_text + ' '.join(answer)))
            print(
                f'Max length {max_seq_length} | Current length {total_tokens + tokens_to_generate} | Haystack: {num_haystack}'
            )
            if total_tokens + tokens_to_generate > max_seq_length:
                num_haystack -= incremental
                break

            if type_haystack == 'essay' and num_haystack > len(haystack):
                num_haystack = len(haystack)
                break

            num_haystack += incremental

        # Generate samples
        for index in range(num_samples):
            used_haystack = num_haystack
            while True:
                try:
                    input_text, answer = _generate_input_output(
                        used_haystack, type_needle_v, template)
                    length = len(
                        tokenizer.encode(input_text)) + tokens_to_generate
                    assert length <= max_seq_length, f'{length} exceeds max_seq_length.'
                    break
                except:
                    if used_haystack > incremental:
                        used_haystack -= incremental

            if remove_newline_tab:
                input_text = ' '.join(
                    input_text.replace('\n', ' ').replace('\t',
                                                          ' ').strip().split())

            data['prompt'].append(input_text)
            data['answer'].append(answer)

        dataset = Dataset.from_dict({
            'prompt': data['prompt'],
            'answer': data['answer'],
        })
        return dataset


class RulerNiahEvaluator(BaseEvaluator):

    def score(self, predictions, gold):
        score = (sum([
            sum([1.0 if r.lower() in pred.lower() else 0.0
                 for r in ref]) / len(ref)
            for pred, ref in zip(predictions, gold)
        ]) / len(predictions) * 100)
        result = {'score': round(score, 2)}
        return result
