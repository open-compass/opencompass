import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import collect_code_preds


class TestCollectCodePreds(unittest.TestCase):

    def test_collect_preds_returns_golds(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            filename = Path(tmp_dir) / 'humanevalx-cpp.json'
            filename.write_text(json.dumps({
                '0': {
                    'origin_prompt': 'Complete the function.',
                    'prediction': 'int add(int a, int b) { return a + b; }',
                    'gold': 'int add(int a, int b) {',
                }
            }),
                                encoding='utf-8')

            succeed, ori_prompts, predictions, golds = (
                collect_code_preds.collect_preds(str(filename)))

            self.assertEqual(succeed, collect_code_preds.SUCCEED)
            self.assertEqual(ori_prompts, ['Complete the function.'])
            self.assertEqual(predictions,
                             ['int add(int a, int b) { return a + b; }'])
            self.assertEqual(golds, ['int add(int a, int b) {'])

    def test_main_passes_gold_to_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            work_dir = tmp_path / 'outputs'
            timestamp = '20260101_000000'
            pred_dir = work_dir / timestamp / 'predictions' / 'test-model'
            pred_dir.mkdir(parents=True)
            (pred_dir / 'humanevalx-cpp.json').write_text(json.dumps({
                '0': {
                    'origin_prompt': 'Complete the function.',
                    'prediction': 'int add(int a, int b) { return a + b; }',
                    'gold': 'int add(int a, int b) {',
                }
            }),
                                                          encoding='utf-8')
            config_path = tmp_path / 'config.py'
            config_path.write_text(
                f'work_dir = {str(work_dir)!r}\n'
                "models = [dict(abbr='test-model')]\n"
                "datasets = [dict(abbr='humanevalx-cpp')]\n",
                encoding='utf-8')

            cleanup_calls = []

            def fake_cleanup(pred, lang, gold):
                cleanup_calls.append((pred, lang, gold))
                return 'cleaned code'

            argv = [
                'collect_code_preds.py',
                str(config_path),
                '-r',
                timestamp,
            ]
            with patch.object(collect_code_preds,
                              '_clean_up_code',
                              side_effect=fake_cleanup), \
                    patch.object(sys, 'argv', argv):
                collect_code_preds.main()

            self.assertEqual(cleanup_calls, [(
                'int add(int a, int b) { return a + b; }',
                'cpp',
                'int add(int a, int b) {',
            )])
            result_file = (work_dir / timestamp / 'humanevalx' / 'test-model' /
                           'humanevalx_cpp.json')
            self.assertEqual(
                result_file.read_text(encoding='utf-8'),
                json.dumps({
                    'task_id': 'CPP/0',
                    'generation': 'cleaned code'
                }) + '\n')


if __name__ == '__main__':
    unittest.main()
