import tempfile
import unittest
from pathlib import Path

from opencompass.openicl.icl_inferencer import BaseInferencer
from opencompass.openicl.icl_inferencer.icl_base_inferencer import (
    ChatOutputHandler, GenInferencerOutputHandler)


class TestBaseInferencer(unittest.TestCase):

    def test_get_dataloader_with_none_batch_size_keeps_batch(self):
        prompt = [{'role': 'user', 'content': 'q1'}]
        gold = {'capability': 'writing'}
        sample = (prompt, gold)

        dataloader = BaseInferencer.get_dataloader([sample], batch_size=None)

        self.assertEqual(next(iter(dataloader)), [sample])


class TestInferencerOutputHandler(unittest.TestCase):

    def test_restore_jsonl_with_unicode_line_separators(self):
        content = 'before\u0085middle\u2028after'
        origin_prompt = [dict(role='user', content=content)]

        for handler_cls in (GenInferencerOutputHandler, ChatOutputHandler):
            with self.subTest(handler=handler_cls.__name__), \
                    tempfile.TemporaryDirectory() as temp_dir:
                writer = handler_cls()
                writer.save_results(origin_prompt, 'answer', 0, gold='gold')
                writer.write_to_jsonl(temp_dir, 'tmp_predictions.jsonl')

                handler = handler_cls()
                restored = handler.restore_from_jsonl(temp_dir,
                                                      'tmp_predictions.jsonl')

                self.assertEqual(restored['0']['origin_prompt'], origin_prompt)
                self.assertEqual(restored['0']['prediction'], 'answer')
                self.assertEqual(restored['0']['gold'], 'gold')
                self.assertEqual(handler.dumped_indices, {'0'})
                self.assertTrue(
                    Path(temp_dir, 'tmp_predictions.jsonl').exists())
                self.assertFalse(
                    Path(temp_dir, 'tmp_predictions.jsonl.bak').exists())
