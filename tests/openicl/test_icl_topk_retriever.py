import unittest
from unittest.mock import patch

import torch
from transformers import BatchEncoding

from opencompass.openicl.icl_retriever.icl_topk_retriever import \
    DataCollatorWithPaddingAndCuda


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


if __name__ == '__main__':
    unittest.main()
