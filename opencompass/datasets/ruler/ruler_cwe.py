# flake8: noqa: F401, E501
import random

import numpy as np
import tiktoken
from datasets import Dataset
from transformers import AutoTokenizer

from opencompass.datasets.base import BaseDataset
from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET


@LOAD_DATASET.register_module()
class RulerCweDataset(BaseDataset):

    @staticmethod
    def load(
        max_seq_length: int = 4096,
        tokenizer_model: str = 'gpt-4',
        template:
        str = 'Below is a numbered list of words. In these words, some appear more often than others. Memorize the ones that appear most often.\n{context}\nQuestion: What are the 10 most common words in the above list? Answer: The top 10 words that appear most often in the list are:',
        tokens_to_generate: int = 120,
        freq_cw: int = 30,
        freq_ucw: int = 3,
        num_cw: int = 10,
        num_samples: int = 500,
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
        try:
            import wonderwords
        except ImportError:
            raise ImportError('''Please install wonderwords by:
                              pip install wonderwords''')
        nouns = wonderwords.random_word._get_words_from_text_file(
            'nounlist.txt')
        adjs = wonderwords.random_word._get_words_from_text_file(
            'adjectivelist.txt')
        verbs = wonderwords.random_word._get_words_from_text_file(
            'verblist.txt')
        words = nouns + adjs + verbs
        words = sorted(list(set(words)))
        random.Random(random_seed).shuffle(words)

        def _get_example(num_words,
                         common_repeats=30,
                         uncommon_repeats=3,
                         common_nums=10):
            word_list_full = random.sample(words, num_words)
            common, uncommon = (
                word_list_full[:common_nums],
                word_list_full[common_nums:],
            )
            word_list = common * int(common_repeats) + uncommon * int(
                uncommon_repeats)
            random.Random(random_seed).shuffle(word_list)

            # Formatting the word list as "1. word1 2. word2 3. word3 ..."
            context = ' '.join(
                [f'{i + 1}. {word}' for i, word in enumerate(word_list)])

            return context, common

        def _generate_input_output(num_words):
            if max_seq_length < 4096:
                context_example, answer_example = _get_example(
                    20, 3, 1, num_cw)
                context, answer = _get_example(num_words, 6, 1, num_cw)
            else:
                context_example, answer_example = _get_example(
                    40, 10, 3, num_cw)
                context, answer = _get_example(num_words, freq_cw, freq_ucw,
                                               num_cw)

            input_example = template.format(
                context=context_example,
                query='',
            ) + ' '.join(
                [f'{i + 1}. {word}' for i, word in enumerate(answer_example)])

            input_text = template.format(
                context=context,
                query='',
            )

            return input_example + '\n' + input_text, answer

        def _sys_word_pair_random(num_samples: int,
                                  max_seq_length: int,
                                  incremental: int = 10):
            data = {'prompt': [], 'answer': []}

            # Find the perfect num_words
            num_words = incremental

            total_tokens = 0
            while total_tokens + tokens_to_generate < max_seq_length:

                input_text, answer = _generate_input_output(num_words)
                # Calculate the number of tokens in the example
                total_tokens = len(
                    tokenizer.encode(input_text + ' ' + ' '.join(
                        [f'{i + 1}. {word}'
                         for i, word in enumerate(answer)])))
                print(
                    f'Max length {max_seq_length} | Current length {total_tokens + tokens_to_generate} | Words: {num_words}'
                )
                if total_tokens + tokens_to_generate > max_seq_length:
                    num_words -= incremental
                    break

                num_words += incremental
                if num_words > len(words):
                    num_words = len(words)
                    break

            print('num_words:', num_words)

            # Generate samples
            for index in range(num_samples):
                used_words = num_words
                while True:
                    try:
                        input_text, answer = _generate_input_output(used_words)
                        length = len(
                            tokenizer.encode(input_text)) + tokens_to_generate
                        assert (length <= max_seq_length
                                ), f'{length} exceeds max_seq_length.'
                        break
                    except:
                        if used_words > incremental:
                            used_words -= incremental

                if remove_newline_tab:
                    input_text = ' '.join(
                        input_text.replace('\n',
                                           ' ').replace('\t',
                                                        ' ').strip().split())

                data['prompt'].append(input_text)
                data['answer'].append(answer)

            return data

        # Generate Data
        data = _sys_word_pair_random(num_samples=num_samples,
                                     max_seq_length=max_seq_length)
        dataset = Dataset.from_dict(data)
        return dataset


class RulerCweEvaluator(BaseEvaluator):

    def score(self, predictions, gold):
        score = (sum([
            sum([1.0 if r.lower() in pred.lower() else 0.0
                 for r in ref]) / len(ref)
            for pred, ref in zip(predictions, gold)
        ]) / len(predictions) * 100)
        result = {'score': round(score, 2)}
        return result
