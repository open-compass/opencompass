import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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

    def test_gen_checkpoint_lookup_cost_grows_linearly(self):

        class CountingSet(set):
            checks = 0

            def __contains__(self, key):
                self.checks += 1
                return super().__contains__(key)

        with tempfile.TemporaryDirectory() as temp_dir:
            handler = GenInferencerOutputHandler()
            handler.dumped_indices = CountingSet()
            num_samples = 64
            for idx in range(num_samples):
                handler.save_results('prompt', str(idx), idx)
                handler.write_to_jsonl(temp_dir, 'tmp.jsonl')
            self.assertLessEqual(handler.dumped_indices.checks,
                                 2 * num_samples)

    def test_gen_checkpoint_resume_and_out_of_order_completions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir, 'tmp.jsonl')
            # A checkpoint from an earlier run, in completion order.
            prefix = (b'{"idx":"8","prediction":"eight"}\n'
                      b'{"idx":"2","prediction":"two"}\n')
            path.write_bytes(prefix)
            handler = GenInferencerOutputHandler()
            handler.save_results('q', 'previous run', 1)
            handler.write_to_jsonl(temp_dir, 'previous.jsonl')
            handler.restore_from_jsonl(temp_dir, path.name)
            handler.save_results('q', 'again', 8)
            handler.write_to_jsonl(temp_dir, path.name)
            self.assertEqual(path.read_bytes(), prefix)

            handler.save_results('q', 'draft', 5)
            handler.save_results('q', 'one', 1)
            handler.save_results('q',
                                 'five',
                                 5,
                                 gold='五',
                                 res_length=1,
                                 input_length=2)
            handler.write_to_jsonl(temp_dir, path.name)
            handler.write_to_jsonl(temp_dir, path.name)
            handler.save_results('q', 'three', 3)
            handler.write_to_jsonl(temp_dir, path.name)
            self.assertTrue(path.read_bytes().startswith(prefix))
            with path.open(encoding='utf-8') as f:
                rows = [json.loads(line) for line in f]
            self.assertEqual([row['idx'] for row in rows],
                             ['8', '2', '5', '1', '3'])
            self.assertEqual(
                rows[2],
                dict(idx='5',
                     origin_prompt='q',
                     prediction='five',
                     gold='五',
                     res_length=1,
                     all_input_length=2))
            handler.write_to_json(temp_dir, 'predictions.json')
            final_path = Path(temp_dir, 'predictions.json')
            final = json.loads(final_path.read_text(encoding='utf-8'))
            self.assertEqual(final['8']['prediction'], 'again')
            self.assertEqual(set(final), {'8', '2', '5', '1', '3'})

    def test_gen_checkpoint_serialization_failure_can_be_retried(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir, 'tmp.jsonl')
            handler = GenInferencerOutputHandler()
            handler.save_results('q', 'valid', 2)
            handler.save_results('q', object(), 1)
            with self.assertRaises(TypeError):
                handler.write_to_jsonl(temp_dir, path.name)
            handler.save_results('q', 'fixed', 1)
            handler.write_to_jsonl(temp_dir, path.name)
            with path.open(encoding='utf-8') as f:
                rows = [json.loads(line) for line in f]
            self.assertEqual([row['prediction'] for row in rows],
                             ['valid', 'fixed'])

    def test_gen_checkpoint_partial_append_can_be_retried(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir, 'tmp.jsonl')
            handler = GenInferencerOutputHandler()
            handler.save_results('q', 'first', 0)
            handler.write_to_jsonl(temp_dir, path.name)
            prefix = path.read_bytes()
            handler.save_results('q', 'second', 1)
            with path.open('a', encoding='utf-8') as stream:
                write = stream.write

                def partial_write(data):
                    write(data[:5])
                    raise OSError('interrupted write')

                with patch('builtins.open', return_value=stream), \
                        patch.object(stream, 'write', partial_write), \
                        self.assertRaises(OSError):
                    handler.write_to_jsonl(temp_dir, path.name)
            self.assertEqual(path.read_bytes(), prefix)
            handler.write_to_jsonl(temp_dir, path.name)
            with path.open(encoding='utf-8') as f:
                self.assertEqual([json.loads(line)['idx'] for line in f],
                                 ['0', '1'])

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
