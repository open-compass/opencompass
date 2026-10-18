import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from transformers import BatchEncoding

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever.icl_topk_retriever import (
    DataCollatorWithPaddingAndCuda, TopkRetriever)

try:
    import faiss
except ImportError:
    faiss = None


class _Tokenizer:

    def pad(self, features, **kwargs):
        return BatchEncoding({
            'input_ids':
            torch.tensor([x['input_ids'] for x in features]),
            'attention_mask':
            torch.tensor([x['attention_mask'] for x in features])
        })


def _all_values_to(batch, device):
    batch.data = {key: value.to(device) for key, value in batch.data.items()}
    return batch


def _tensor_only_to(batch, device):
    batch.data = {
        key: value.to(device)
        for key, value in batch.data.items()
        if isinstance(value, torch.Tensor)
    }
    return batch


class TestDataCollatorWithPaddingAndCuda(unittest.TestCase):

    def test_preserves_metadata_across_batch_encoding_to_behaviors(self):
        metadata = [{
            'id': 0,
            'len': 2,
            'text': 'first'
        }, {
            'id': 1,
            'len': 2,
            'text': 'second'
        }]

        for to_method in (_all_values_to, _tensor_only_to):
            with self.subTest(to_method=to_method.__name__), \
                    patch.object(BatchEncoding, 'to', to_method):
                features = [{
                    'input_ids': [1, 2],
                    'attention_mask': [1, 1],
                    'metadata': metadata[0]
                }, {
                    'input_ids': [3, 4],
                    'attention_mask': [1, 1],
                    'metadata': metadata[1]
                }]

                batch = DataCollatorWithPaddingAndCuda(_Tokenizer(),
                                                       device='cpu')(features)

                self.assertIsInstance(batch, BatchEncoding)
                torch.testing.assert_close(batch['input_ids'],
                                           torch.tensor([[1, 2], [3, 4]]))
                torch.testing.assert_close(batch['attention_mask'],
                                           torch.tensor([[1, 1], [1, 1]]))
                self.assertEqual(batch['metadata'], metadata)


@unittest.skipIf(faiss is None, 'faiss is required for nearest-neighbor tests')
class TestTopkRetriever(unittest.TestCase):

    def setUp(self):
        self.retriever = object.__new__(TopkRetriever)
        self.retriever.index = faiss.IndexIDMap(faiss.IndexFlatIP(2))
        self.retriever.index.add_with_ids(
            np.array([[1, 0], [0, 1]], dtype=np.float32),
            np.array([0, 1], dtype=np.int64))
        self.retriever.dataloader = None
        self.retriever.is_main_process = False
        self.retriever.ice_num = 4
        self.embeddings = [{
            'embed': np.array([0, 1], dtype=np.float32),
            'metadata': {
                'id': 0
            }
        }]

    def test_retrieve_excludes_missing_neighbors(self):
        with patch.object(self.retriever,
                          'forward',
                          return_value=self.embeddings):
            indices = self.retriever.retrieve()

        self.assertEqual(indices, [[1, 0]])

    def test_missing_neighbors_do_not_duplicate_last_example_in_prompt(self):
        self.retriever.index_ds = [{
            'question': 'first',
            'answer': 'A'
        }, {
            'question': 'second',
            'answer': 'B'
        }]
        self.retriever.dataset_reader = SimpleNamespace(output_column='answer')
        self.retriever.ice_separator = '\n'
        self.retriever.ice_eos_token = '\n'
        template = PromptTemplate('{question}: {answer}')

        with patch.object(self.retriever,
                          'forward',
                          return_value=self.embeddings):
            indices = self.retriever.retrieve()[0]

        self.assertEqual(self.retriever.generate_ice(indices, template),
                         'second: B\nfirst: A\n')

    def test_retrieve_preserves_rank_for_available_neighbors(self):
        for count in (1, 2):
            with self.subTest(count=count), patch.object(
                    self.retriever, 'forward', return_value=self.embeddings):
                self.retriever.ice_num = count
                self.assertEqual(self.retriever.retrieve(), [[1, 0][:count]])


if __name__ == '__main__':
    unittest.main()
