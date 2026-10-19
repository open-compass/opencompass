# flake8: noqa: F401, E501
import random
import string

import numpy as np
import tiktoken
from datasets import Dataset
from scipy.special import zeta
from transformers import AutoTokenizer

from opencompass.datasets.base import BaseDataset
from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET


@LOAD_DATASET.register_module()
class RulerFweDataset(BaseDataset):

    @staticmethod
    def load(
        max_seq_length: int = 4096,
        tokenizer_model: str = 'gpt-4',
        template:
        str = "Read the following coded text and track the frequency of each coded word. Find the three most frequently appeared coded words. {context}\nQuestion: Do not provide any explanation. Please ignore the dots '....'. What are the three most frequently appeared words in the above coded text? Answer: According to the coded text above, the three most frequently appeared words are:",
        tokens_to_generate: int = 50,
        alpha: float = 2.0,
        coded_wordlen: int = 6,
        num_samples: int = 500,
        random_seed: int = 42,
        remove_newline_tab: str = '',
        vocab_size: int = -1,
    ) -> Dataset:

        if tokenizer_model == 'gpt-4':
            tokenizer = tiktoken.encoding_for_model(tokenizer_model)
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_model,
                                                      trust_remote_code=True)

        random.seed(random_seed)
        np.random.seed(random_seed)

        def _generate_input_output(
            max_len,
            num_words=-1,
            coded_wordlen=6,
            vocab_size=2000,
            incremental=10,
            alpha=2.0,
        ):
            # generate vocab
            vocab = [
                ''.join(random.choices(string.ascii_lowercase,
                                       k=coded_wordlen))
                for _ in range(vocab_size)
            ]
            while len(set(vocab)) < vocab_size:
                vocab.append(''.join(
                    random.choices(string.ascii_lowercase, k=coded_wordlen)))
            vocab = sorted(list(set(vocab)))
            random.Random(random_seed).shuffle(vocab)
            vocab[0] = '...'  # treat the top ranked as noise

            # sample words
            def gen_text(num_words):
                k = np.arange(1, len(vocab) + 1)
                sampled_cnt = num_words * (k**-alpha) / zeta(alpha)
                sampled_words = [
                    [w] * zi for w, zi in zip(vocab, sampled_cnt.astype(int))
                ]
                sampled_words = [x for wlst in sampled_words for x in wlst]
                random.Random(random_seed).shuffle(sampled_words)
                return (
                    template.format(context=' '.join(sampled_words), query=''),
                    vocab[1:4],
                )

            if num_words > 0:
                num_words = num_words
                text, answer = gen_text(num_words)
                while len(tokenizer.encode(text)) > max_len:
                    num_words -= incremental
                    text, answer = gen_text(num_words)
            else:
                num_words = max_len // coded_wordlen  # init
                text, answer = gen_text(num_words)
                while len(tokenizer.encode(text)) < max_len:
                    num_words += incremental
                    text, answer = gen_text(num_words)
                num_words -= incremental
            text, answer = gen_text(num_words)
            return text, answer, num_words

        def _sys_kwext(
            num_samples: int,
            max_seq_length: int,
            vocab_size: int = -1,
            incremental: int = 10,
        ):
            data = {'prompt': [], 'answer': []}

            vocab_size = max_seq_length // 50 if vocab_size == -1 else vocab_size

            # get number of words
            input_max_len = max_seq_length
            _, _, num_example_words = _generate_input_output(
                input_max_len,
                coded_wordlen=coded_wordlen,
                vocab_size=vocab_size,
                incremental=input_max_len // 32,
                alpha=alpha,
            )
            print('num_example_words:', num_example_words)
            # Generate samples
            for index in range(num_samples):

                # construct input
                input_max_len = max_seq_length
                input_text, answer, _ = _generate_input_output(
                    input_max_len,
                    num_words=num_example_words,
                    coded_wordlen=coded_wordlen,
                    vocab_size=vocab_size,
                    incremental=input_max_len // 32,
                    alpha=alpha,
                )

                length = len(tokenizer.encode(input_text)) + tokens_to_generate

                if remove_newline_tab:
                    input_text = ' '.join(
                        input_text.replace('\n',
                                           ' ').replace('\t',
                                                        ' ').strip().split())

                data['prompt'].append(input_text)
                data['answer'].append(answer)

            return data

        # Generate Data
        data = _sys_kwext(
            num_samples=num_samples,
            max_seq_length=max_seq_length,
            vocab_size=vocab_size,
            incremental=10,
        )
        dataset = Dataset.from_dict(data)
        return dataset


class RulerFweEvaluator(BaseEvaluator):

    def score(self, predictions, gold):
        score = (sum([
            sum([1.0 if r.lower() in pred.lower() else 0.0
                 for r in ref]) / len(ref)
            for pred, ref in zip(predictions, gold)
        ]) / len(predictions) * 100)
        result = {'score': round(score, 2)}
        return result
