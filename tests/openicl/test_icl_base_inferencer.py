import unittest

from opencompass.openicl.icl_inferencer import BaseInferencer


class TestBaseInferencer(unittest.TestCase):

    def test_get_dataloader_with_none_batch_size_keeps_batch(self):
        prompt = [{'role': 'user', 'content': 'q1'}]
        gold = {'capability': 'writing'}
        sample = (prompt, gold)

        dataloader = BaseInferencer.get_dataloader([sample], batch_size=None)

        self.assertEqual(next(iter(dataloader)), [sample])
