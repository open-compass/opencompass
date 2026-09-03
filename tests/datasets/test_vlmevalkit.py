import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from datasets import Dataset
from mmengine.config import Config

from opencompass.datasets.vlmevalkit import (VLMEvalKitDataset,
                                             build_vlmeval_dataset,
                                             convert_vlmeval_prompt)
from opencompass.datasets.vlmevalkit.builder import _load_build_dataset
from opencompass.evaluator.vlmevalkit import VLMEvalKitEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.tasks import OpenICLEvalTask


class FakeVLMEvalDataset:
    MODALITY = 'IMAGE'
    TYPE = 'MCQ'

    def __init__(self):
        self.data = pd.DataFrame([
            dict(index=101,
                 question='first question',
                 hint='first hint',
                 A='one',
                 B='two',
                 answer='A',
                 split='dev',
                 category='logic',
                 **{'l2-category': 'reasoning'},
                 image='base64-1'),
            dict(index=202,
                 question='second question',
                 hint=None,
                 A='three',
                 B='four',
                 answer='B',
                 split='dev',
                 category='perception',
                 **{'l2-category': 'vision'},
                 image='base64-2'),
        ])
        self.build_prompt_calls = []
        self.build_prompt_roots = []
        self.evaluate_calls = []
        self.evaluate_roots = []

    def build_prompt(self, row):
        self.build_prompt_calls.append(row['index'])
        self.build_prompt_roots.append(os.environ.get('LMUData'))
        return [
            dict(type='image', value=f"/images/{row['index']}.jpg"),
            dict(type='text',
                 value=f"Question: {row['question']}\n"
                 f"Options:\nA. {row['A']}\nB. {row['B']}")
        ]

    def evaluate(self, result_file, **kwargs):
        self.evaluate_calls.append((result_file, kwargs))
        self.evaluate_roots.append(os.environ.get('LMUData'))
        Path(result_file).with_name('MMBench_DEV_EN_acc.csv').write_text(
            'split,Overall\ndev,0.5\n', encoding='utf-8')
        return pd.DataFrame([dict(split='dev', Overall=0.5, logic=1.0)])

    @classmethod
    def report_primary_metric(cls, metrics):
        return {'split=dev|Overall': metrics['split=dev|Overall']}

    def infer_data_job(self, *args, **kwargs):
        raise AssertionError('VLMEvalKit inference must not be called')

    def run(self, *args, **kwargs):
        raise AssertionError('VLMEvalKit runner must not be called')


class FakeMMMUProDataset:
    MODALITY = 'IMAGE'
    TYPE = 'MCQ_MMMU_Pro'

    def __init__(self):
        self.data = pd.DataFrame([
            dict(index=7,
                 id='history-7',
                 question='before <image 1> middle <image 2> after',
                 A='alpha',
                 B='beta',
                 answer='B',
                 category='History',
                 topic_difficulty='hard',
                 image=['base64-a', 'base64-b'],
                 image_path=['a.png', 'b.png']),
            dict(index=9,
                 id='art-9',
                 question='question <image 1>',
                 A='alpha',
                 B='beta',
                 answer='A',
                 category='Art',
                 topic_difficulty='easy',
                 image=['base64-c'],
                 image_path='c.png'),
        ])
        self.build_prompt_calls = []
        self.evaluate_calls = []

    def build_prompt(self, row):
        self.build_prompt_calls.append(row['index'])
        if row['index'] == 7:
            return [
                dict(type='text', value='Question: before '),
                dict(type='image', value='/images/a.png'),
                dict(type='text', value=' middle '),
                dict(type='image', value='/images/b.png'),
                dict(type='text',
                     value=' after\nOptions:\nA. alpha\nB. beta\n')
            ]
        return [
            dict(type='text', value='Question: question '),
            dict(type='image', value='/images/c.png'),
            dict(type='text', value='\nOptions:\nA. alpha\nB. beta\n')
        ]

    def evaluate(self, result_file, **kwargs):
        self.evaluate_calls.append((result_file, kwargs))
        Path(result_file).with_name('MMMU_Pro_10c_acc.csv').write_text(
            'split,Overall,Art,History\nnone,0.5,0.0,1.0\n', encoding='utf-8')
        return pd.DataFrame(
            [dict(split='none', Overall=0.5, Art=0.0, History=1.0)])

    @classmethod
    def report_primary_metric(cls, metrics):
        return {'Overall Acc': metrics['split=none|Overall'] * 100}


def fake_flatten_metrics(raw_metrics):
    metrics = {}
    for row_index, row in raw_metrics.iterrows():
        dimensions = []
        for key, value in row.items():
            if not isinstance(value, (int, float)):
                dimensions.append(f'{key}={value}')
        prefix = '|'.join(dimensions)
        for key, value in row.items():
            if isinstance(value, (int, float)):
                metrics[f'{prefix}|{key}' if prefix else key] = value
    return metrics


class FakeModel:
    is_api = True
    generation_kwargs = {}

    def __init__(self):
        self.inputs = []

    def generate(self, inputs, max_out_len):
        return []

    def parse_template(self, entries, mode='gen'):
        return entries

    def generate_from_template(self, entries, max_out_len, **kwargs):
        self.inputs.extend(entries)
        return [
            'A' if 'first question' in str(entry) else 'B' for entry in entries
        ]

    def get_token_len(self, value):
        return len(str(value))


class TestVLMEvalKitDataset(unittest.TestCase):

    def test_full_config_loads_without_importing_vlmeval(self):
        path = (Path(__file__).resolve().parents[2] / 'examples' /
                'eval_mmbench_vlmevalkit.py')
        config = Config.fromfile(path)
        self.assertEqual(config.datasets[0]['abbr'], 'MMBench_DEV_EN')
        self.assertIsNone(config.datasets[0]['sample_limit'])
        self.assertIsNone(
            config.datasets[0]['eval_cfg']['evaluator']['sample_limit'])
        self.assertEqual(config.models[0]['type'].__name__, 'OpenAISDK')
        self.assertEqual(config.models[0]['abbr'],
                         'kimi-k2.6-chat-completions')
        self.assertEqual(config.models[0]['path'], 'kimi-k2.6')
        self.assertEqual(config.models[0]['key'], 'ENV')
        self.assertEqual(config.models[0]['openai_api_base'],
                         'https://example.com/v1')
        self.assertEqual(config.models[0]['max_out_len'], 32768)
        self.assertEqual(config.models[0]['batch_size'], 64)
        self.assertEqual(config.models[0]['max_workers'], 4)
        self.assertEqual(config.models[0]['query_per_second'], 3)
        self.assertEqual(config.models[0]['timeout'], 3600)
        judge = config.datasets[0]['eval_cfg']['evaluator']['eval_kwargs']
        self.assertEqual(config.vlmeval_eval_kwargs, judge)
        self.assertEqual(
            judge,
            dict(model='kimi-k2.6',
                 api_base='https://example.com/v1/chat/completions',
                 nproc=4,
                 retry=3,
                 timeout=600,
                 temperature=0.0,
                 max_tokens=32768))

    def test_dataset_config_has_no_concrete_judge(self):
        path = (Path(__file__).resolve().parents[2] / 'opencompass' /
                'configs' / 'datasets' / 'MMBench' /
                'MMBench_DEV_EN_vlmevalkit_gen.py')
        config = Config.fromfile(path)

        evaluator = config.mmbench_datasets[0]['eval_cfg']['evaluator']
        self.assertEqual(evaluator['eval_kwargs'], {})
        self.assertEqual(evaluator['dataset_kwargs'], {})

    def test_mmmu_pro_config_loads(self):
        path = (Path(__file__).resolve().parents[2] / 'examples' /
                'eval_mmmu_pro_vlmevalkit.py')
        with patch.dict(os.environ, {'MMMU_PRO_SAMPLE_LIMIT': '4'}):
            config = Config.fromfile(path)

        self.assertEqual(config.datasets[0]['abbr'], 'MMMU_Pro_10c')
        self.assertEqual(config.datasets[0]['sample_limit'], 4)
        evaluator = config.datasets[0]['eval_cfg']['evaluator']
        self.assertEqual(evaluator['dataset_name'], 'MMMU_Pro_10c')
        self.assertEqual(evaluator['sample_limit'], 4)

    def test_examples_use_native_openicl_inference(self):
        examples = Path(__file__).resolve().parents[2] / 'examples'
        mmbench = Config.fromfile(examples / 'eval_mmbench_vlmevalkit.py')
        mmmu_pro = Config.fromfile(examples / 'eval_mmmu_pro_vlmevalkit.py')

        for config in (mmbench, mmmu_pro):
            self.assertEqual(
                config.datasets[0]['infer_cfg']['inferencer']['type'].__name__,
                'GenInferencer')
            self.assertNotIn(
                'prediction_fields',
                config.datasets[0]['infer_cfg']['inferencer'])
            self.assertEqual(config.infer.runner.task.type.__name__,
                             'OpenICLInferTask')
            self.assertEqual(config.eval.runner.task.type.__name__,
                             'OpenICLEvalTask')
        for config in (mmbench, mmmu_pro):
            self.assertNotIn('failure_message', config.models[0])
            self.assertNotIn('skip_failed', config.models[0])

    def test_full_config_applies_sample_limit(self):
        path = (Path(__file__).resolve().parents[2] / 'examples' /
                'eval_mmbench_vlmevalkit.py')
        with patch.dict(os.environ, {'MMBENCH_SAMPLE_LIMIT': '4'}):
            config = Config.fromfile(path)

        self.assertEqual(config.datasets[0]['sample_limit'], 4)
        self.assertEqual(
            config.datasets[0]['eval_cfg']['evaluator']['sample_limit'], 4)

    def test_missing_dependency_error(self):
        error = ModuleNotFoundError("No module named 'vlmeval'",
                                    name='vlmeval')
        with patch('importlib.import_module', side_effect=error):
            with self.assertRaisesRegex(ModuleNotFoundError,
                                        'VLMEvalKit is required'):
                _load_build_dataset()

    def test_incomplete_wheel_error(self):
        error = ModuleNotFoundError(
            "No module named 'vlmeval.dataset.utils.mmhelix'",
            name='vlmeval.dataset.utils.mmhelix')
        with patch('importlib.import_module', side_effect=error):
            with self.assertRaisesRegex(ModuleNotFoundError,
                                        'incomplete or missing optional'):
                _load_build_dataset()

    def test_python_version_error(self):
        with patch('opencompass.datasets.vlmevalkit.builder.sys.version_info',
                   (3, 9)):
            with self.assertRaisesRegex(RuntimeError, 'Python 3.10 or newer'):
                _load_build_dataset()

    def test_build_dataset_import_is_lazy(self):
        builder = object()
        module = type('Module', (), {'build_dataset': builder})()
        with patch('importlib.import_module', return_value=module):
            self.assertIs(_load_build_dataset(), builder)

    def test_dataset_compatibility_uses_upstream_instance_metadata(self):
        fake = FakeMMMUProDataset()
        calls = []

        def build(name, **kwargs):
            calls.append((name, kwargs))
            return fake

        with patch(
                'opencompass.datasets.vlmevalkit.builder.'
                '_load_build_dataset',
                return_value=build):
            self.assertIs(
                build_vlmeval_dataset('New_Upstream_Image_Benchmark',
                                      custom=True), fake)
        self.assertEqual(calls,
                         [('New_Upstream_Image_Benchmark', dict(custom=True))])

        fake.TYPE = 'VQA'
        with patch(
                'opencompass.datasets.vlmevalkit.builder.'
                '_load_build_dataset',
                return_value=lambda name, **kwargs: fake):
            self.assertIs(build_vlmeval_dataset('Another_Image_Benchmark'),
                          fake)

        fake.MODALITY = 'VIDEO'
        with patch(
                'opencompass.datasets.vlmevalkit.builder.'
                '_load_build_dataset',
                return_value=lambda name, **kwargs: fake):
            with self.assertRaisesRegex(NotImplementedError,
                                        'supports IMAGE datasets only'):
                build_vlmeval_dataset('Video-MME')

        fake.MODALITY = 'IMAGE'
        fake.TYPE = 'MT'
        with patch(
                'opencompass.datasets.vlmevalkit.builder.'
                '_load_build_dataset',
                return_value=lambda name, **kwargs: fake):
            with self.assertRaisesRegex(NotImplementedError, 'multi-turn'):
                build_vlmeval_dataset('MMDU')

    def test_official_builder_and_dataset_conversion(self):
        fake = FakeVLMEvalDataset()
        roots = []

        def build_dataset(name, **kwargs):
            roots.append((name, kwargs, os.environ.get('LMUData')))
            return fake

        old_root = os.environ.get('LMUData')
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                    'opencompass.datasets.vlmevalkit.builder.'
                    '_load_build_dataset',
                    return_value=build_dataset):
                first = VLMEvalKitDataset(
                    abbr='MMBench_DEV_EN',
                    dataset_name='MMBench_DEV_EN',
                    data_root=temp_dir,
                    reader_cfg=dict(input_columns=['prompt'],
                                    output_column='answer'))
                second = VLMEvalKitDataset(
                    abbr='MMBench_DEV_EN',
                    dataset_name='MMBench_DEV_EN',
                    data_root=temp_dir,
                    reader_cfg=dict(input_columns=['prompt'],
                                    output_column='answer'))

        self.assertEqual(roots, [('MMBench_DEV_EN', {}, temp_dir),
                                 ('MMBench_DEV_EN', {}, temp_dir)])
        self.assertEqual(os.environ.get('LMUData'), old_root)
        self.assertEqual(fake.build_prompt_calls, [101, 202, 101, 202])
        self.assertEqual(fake.build_prompt_roots, [temp_dir] * 4)
        self.assertEqual(first.test['sample_id'], second.test['sample_id'])
        self.assertEqual(first.test[0]['sample_id'], 'MMBench_DEV_EN:101')
        self.assertEqual(first.test[0]['vlmeval_index'], 101)
        raw_sample = json.loads(first.test[0]['raw_sample'])
        self.assertEqual(raw_sample['question'], 'first question')
        self.assertEqual(raw_sample['A'], 'one')
        self.assertEqual(raw_sample['answer'], 'A')
        self.assertNotIn('image', raw_sample)
        content = first.test[0]['prompt'][0]['content']
        self.assertEqual([item['type'] for item in content], ['image', 'text'])
        self.assertEqual(content[0]['image_url'], '/images/101.jpg')
        self.assertEqual(content[1]['text'],
                         'Question: first question\nOptions:\nA. one\nB. two')
        json.dumps(first.test[0]['prompt'])

    def test_mmmu_pro_multimage_prompt_and_raw_fields(self):
        fake = FakeMMMUProDataset()
        with patch(
                'opencompass.datasets.vlmevalkit.dataset.'
                'build_vlmeval_dataset',
                return_value=fake):
            first = VLMEvalKitDataset(
                abbr='MMMU_Pro_10c',
                dataset_name='MMMU_Pro_10c',
                reader_cfg=dict(input_columns=['prompt'],
                                output_column='answer'))
            second = VLMEvalKitDataset(
                abbr='MMMU_Pro_10c',
                dataset_name='MMMU_Pro_10c',
                reader_cfg=dict(input_columns=['prompt'],
                                output_column='answer'))

        self.assertEqual(fake.build_prompt_calls, [7, 9, 7, 9])
        self.assertEqual(first.test['sample_id'], second.test['sample_id'])
        self.assertEqual(first.test[0]['sample_id'], 'MMMU_Pro_10c:7')
        raw_sample = json.loads(first.test[0]['raw_sample'])
        self.assertEqual(raw_sample['id'], 'history-7')
        self.assertEqual(raw_sample['image_path'], ['a.png', 'b.png'])
        self.assertEqual(raw_sample['topic_difficulty'], 'hard')
        self.assertNotIn('image', raw_sample)
        self.assertEqual(
            json.loads(first.test[1]['raw_sample'])['image_path'], 'c.png')
        content = first.test[0]['prompt'][0]['content']
        self.assertEqual([item['type'] for item in content],
                         ['text', 'image', 'text', 'image', 'text'])
        self.assertEqual(content[1]['image_url'], '/images/a.png')
        self.assertEqual(content[3]['image_url'], '/images/b.png')
        json.dumps(first.test[0]['prompt'])

    def test_sample_limit_builds_only_requested_prompts(self):
        fake = FakeVLMEvalDataset()
        with patch(
                'opencompass.datasets.vlmevalkit.dataset.'
                'build_vlmeval_dataset',
                return_value=fake):
            dataset = VLMEvalKitDataset(
                abbr='MMBench_DEV_EN',
                dataset_name='MMBench_DEV_EN',
                sample_limit=1,
                reader_cfg=dict(input_columns=['prompt'],
                                output_column='answer'))

        self.assertEqual(len(dataset.test), 1)
        self.assertEqual(dataset.test[0]['sample_id'], 'MMBench_DEV_EN:101')
        self.assertEqual(fake.build_prompt_calls, [101])
        self.assertIn('prompt', dataset.test.column_names)

    def test_test_range_builds_only_partition_prompts(self):
        for test_range in ('[1:2]', '[:2][1:2]'):
            with self.subTest(test_range=test_range):
                fake = FakeVLMEvalDataset()
                reader_cfg = dict(input_columns=['prompt'],
                                  output_column='answer',
                                  test_range=test_range)
                with patch(
                        'opencompass.datasets.vlmevalkit.dataset.'
                        'build_vlmeval_dataset',
                        return_value=fake):
                    dataset = VLMEvalKitDataset(
                        abbr='MMBench_DEV_EN_1',
                        dataset_name='MMBench_DEV_EN',
                        reader_cfg=reader_cfg)

                self.assertEqual(reader_cfg['test_range'], test_range)
                self.assertEqual(fake.build_prompt_calls, [202])
                self.assertEqual(dataset.test['vlmeval_index'], [202])
                self.assertIn('prompt', dataset.test.column_names)

    def test_index_contract_fails_fast(self):
        cases = [
            (FakeVLMEvalDataset().data.drop(columns='index'),
             'no `index` column'),
            (FakeVLMEvalDataset().data.assign(index=[101, None]),
             'null `index` values'),
            (FakeVLMEvalDataset().data.assign(index=[101, '101']),
             'duplicate `index` values'),
        ]
        for data, message in cases:
            with self.subTest(message=message):
                fake = FakeVLMEvalDataset()
                fake.data = data
                with patch(
                        'opencompass.datasets.vlmevalkit.builder.'
                        '_load_build_dataset',
                        return_value=lambda name, **kwargs: fake):
                    with self.assertRaisesRegex(ValueError, message):
                        build_vlmeval_dataset('MMBench_DEV_EN')

    def test_prompt_conversion_preserves_role_and_order(self):
        prompt = [
            dict(role='system', type='text', value='system'),
            dict(type='text', value='before'),
            dict(type='image', value='/image.jpg'),
            dict(type='text', value='after'),
        ]
        converted = convert_vlmeval_prompt(prompt)
        self.assertEqual(converted, [
            dict(role='system', content=[dict(type='text', text='system')]),
            dict(role='user',
                 content=[
                     dict(type='text', text='before'),
                     dict(type='image', image_url='/image.jpg'),
                     dict(type='text', text='after')
                 ])
        ])

    def test_prompt_conversion_uses_structure_not_dataset_name(self):
        prompt = [
            dict(role='system', content=[dict(type='text', value='system')]),
            dict(role='user',
                 content=[
                     dict(type='text', value='before'),
                     dict(type='image', value='/image.jpg'),
                     dict(type='text', value='after')
                 ]),
            dict(role='assistant', content=[dict(type='text', value='answer')])
        ]

        self.assertEqual(convert_vlmeval_prompt(prompt), [
            dict(role='system', content=[dict(type='text', text='system')]),
            dict(role='user',
                 content=[
                     dict(type='text', text='before'),
                     dict(type='image', image_url='/image.jpg'),
                     dict(type='text', text='after')
                 ]),
            dict(role='assistant', content=[dict(type='text', text='answer')])
        ])

    def test_unknown_prompt_capability_fails(self):
        with self.assertRaisesRegex(ValueError, "prompt type: 'video'"):
            convert_vlmeval_prompt([dict(type='video', value='/video.mp4')])


class TestVLMEvalKitEvaluator(unittest.TestCase):

    def setUp(self):
        self.fake = FakeVLMEvalDataset()
        self.ids = ['MMBench_DEV_EN:101', 'MMBench_DEV_EN:202']
        metric_patch = patch(
            'opencompass.evaluator.vlmevalkit._flatten_vlmeval_metrics',
            side_effect=fake_flatten_metrics)
        metric_patch.start()
        self.addCleanup(metric_patch.stop)

    def _score(self,
               output_dir,
               predictions=None,
               sample_id=None,
               vlmeval_index=None):
        evaluator = VLMEvalKitEvaluator(dataset_name='MMBench_DEV_EN',
                                        eval_kwargs=dict(
                                            model='exact_matching', nproc=2))
        evaluator._out_dir = output_dir
        predictions = ['A', 'B'] if predictions is None else predictions
        sample_id = self.ids if sample_id is None else sample_id
        vlmeval_index = ([101, 202]
                         if vlmeval_index is None else vlmeval_index)
        test_set = Dataset.from_list([
            dict(sample_id=current_id,
                 vlmeval_index=index,
                 dataset_name='MMBench_DEV_EN')
            for current_id, index in zip(sample_id, vlmeval_index)
        ])
        return evaluator.score(predictions=predictions,
                               test_set=test_set)

    def test_alignment_result_file_evaluate_and_artifacts(self):
        written = []

        def to_excel(frame, path, index=False):
            written.append((frame.copy(), Path(path), index))
            Path(path).touch()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = str(Path(temp_dir) / 'result')
            with patch(
                    'opencompass.evaluator.vlmevalkit.build_vlmeval_dataset',
                    return_value=self.fake), patch.object(pd.DataFrame,
                                                          'to_excel',
                                                          new=to_excel):
                result = self._score(output_dir,
                                     predictions=['B', 'A'],
                                     sample_id=list(reversed(self.ids)),
                                     vlmeval_index=[202, 101])

            frame, result_file, index = written[0]
            self.assertFalse(index)
            self.assertEqual(frame['index'].tolist(), [101, 202])
            self.assertEqual(frame['prediction'].tolist(), ['A', 'B'])
            self.assertNotIn('image', frame)
            self.assertTrue({
                'question', 'A', 'B', 'answer', 'split', 'category',
                'l2-category'
            }.issubset(frame.columns))
            self.assertEqual(result_file.name, 'MMBench_DEV_EN.xlsx')
            self.assertEqual(
                self.fake.evaluate_calls,
                [(str(result_file), dict(model='exact_matching', nproc=2))])
            self.assertEqual(result['split=dev|Overall'], 50.0)
            self.assertEqual(result['split=dev|logic'], 100.0)
            self.assertTrue(
                (Path(output_dir) / 'MMBench_DEV_EN_acc.csv').exists())
            config = json.loads(
                (Path(output_dir) /
                 'vlmevalkit_evaluation.json').read_text(encoding='utf-8'))
            self.assertEqual(config['eval_kwargs']['model'], 'exact_matching')
            raw_metrics = json.loads(
                (Path(output_dir) /
                 'vlmevalkit_metrics.json').read_text(encoding='utf-8'))
            self.assertEqual(raw_metrics['flattened']['split=dev|Overall'],
                             0.5)

    def test_invalid_prediction_alignment_fails(self):
        cases = [
            (['A'], self.ids, [101, 202],
             'predictions and test set have different lengths'),
            (['A',
              'A'], [self.ids[0],
                     self.ids[0]], [101,
                                    101], 'Duplicate VLMEvalKit sample_id'),
            (['A', 'B'], [self.ids[0], 'MMBench_DEV_EN:999'], [101, 999],
             'Unknown VLMEvalKit sample_id'),
            ([None, 'B'], self.ids, [101,
                                     202], 'Missing VLMEvalKit prediction'),
            ([' ', 'B'], self.ids, [101,
                                    202], 'Missing VLMEvalKit prediction'),
            (['A', 'B'], self.ids, [999, 202], 'VLMEvalKit index mismatch'),
        ]
        with tempfile.TemporaryDirectory() as temp_dir, patch(
                'opencompass.evaluator.vlmevalkit.build_vlmeval_dataset',
                return_value=self.fake):
            for predictions, sample_ids, indexes, message in cases:
                with self.subTest(message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        self._score(temp_dir, predictions, sample_ids, indexes)

    def test_sample_limit_evaluates_matching_subset(self):
        evaluator = VLMEvalKitEvaluator(
            dataset_name='MMBench_DEV_EN',
            sample_limit=1,
            eval_kwargs=dict(model='exact_matching'))
        written = []

        def to_excel(frame, path, index=False):
            written.append(frame.copy())
            Path(path).touch()

        with tempfile.TemporaryDirectory() as temp_dir, patch(
                'opencompass.evaluator.vlmevalkit.build_vlmeval_dataset',
                return_value=self.fake), patch.object(pd.DataFrame,
                                                      'to_excel',
                                                      new=to_excel):
            evaluator._out_dir = temp_dir
            evaluator.score(predictions=['A'],
                            test_set=Dataset.from_list([
                                dict(sample_id=self.ids[0],
                                     vlmeval_index=101,
                                     dataset_name='MMBench_DEV_EN')
                            ]))

            config = json.loads(
                (Path(temp_dir) / 'vlmevalkit_evaluation.json').read_text())

        self.assertEqual(written[0]['index'].tolist(), [101])
        self.assertEqual(written[0]['prediction'].tolist(), ['A'])
        self.assertEqual(config['sample_limit'], 1)

    def test_evaluate_uses_configured_lmu_data_root(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch(
                'opencompass.evaluator.vlmevalkit.build_vlmeval_dataset',
                return_value=self.fake), patch.object(pd.DataFrame,
                                                      'to_excel'):
            evaluator = VLMEvalKitEvaluator(
                dataset_name='MMBench_DEV_EN',
                data_root=temp_dir,
                eval_kwargs=dict(model='exact_matching'))
            evaluator._out_dir = str(Path(temp_dir) / 'result')
            evaluator.score(predictions=['A', 'B'],
                            test_set=Dataset.from_list([
                                dict(sample_id=self.ids[0],
                                     vlmeval_index=101,
                                     dataset_name='MMBench_DEV_EN'),
                                dict(sample_id=self.ids[1],
                                     vlmeval_index=202,
                                     dataset_name='MMBench_DEV_EN')
                            ]))

        self.assertEqual(self.fake.evaluate_roots, [temp_dir])


@unittest.skipUnless(importlib.util.find_spec('vlmeval'),
                     'VLMEvalKit optional dependency is not installed')
class TestVLMEvalKitUpstreamParity(unittest.TestCase):

    @staticmethod
    def _dataset(dataset_name):
        if dataset_name == 'MMBench_DEV_EN':
            from vlmeval.dataset.image_mcq import ImageMCQDataset

            frame = pd.DataFrame([
                dict(index=101,
                     question='first question',
                     A='one',
                     B='two',
                     answer='A',
                     split='dev',
                     category='logic',
                     **{'l2-category': 'reasoning'},
                     image='base64-1',
                     image_path='first.png'),
                dict(index=202,
                     question='second question',
                     A='three',
                     B='four',
                     answer='B',
                     split='dev',
                     category='perception',
                     **{'l2-category': 'vision'},
                     image='base64-2',
                     image_path='second.png'),
            ])
            dataset = object.__new__(ImageMCQDataset)
        else:
            from vlmeval.dataset.image_mcq import MMMUProDataset

            frame = FakeMMMUProDataset().data
            dataset = object.__new__(MMMUProDataset)
        dataset.dataset_name = dataset_name
        dataset.data = frame.copy()
        dataset.meta_only = True
        return dataset

    def test_real_mmmu_pro_prompt_contract(self):
        dataset = self._dataset('MMMU_Pro_10c')
        prompt = dataset.build_prompt(dataset.data.iloc[0])
        converted = convert_vlmeval_prompt(prompt)

        self.assertEqual([item['type'] for item in prompt],
                         ['text', 'image', 'text', 'image', 'text'])
        self.assertEqual([item['type'] for item in converted[0]['content']],
                         ['text', 'image', 'text', 'image', 'text'])
        self.assertEqual(converted[0]['content'][1]['image_url'], 'a.png')
        self.assertIn('A. alpha', converted[0]['content'][-1]['text'])
        self.assertIn('B. beta', converted[0]['content'][-1]['text'])

        for dataset_name in ('MMMU_Pro_V', 'MMMU_Pro_V_COT'):
            dataset = self._dataset(dataset_name)
            prompt = dataset.build_prompt(dataset.data.iloc[1])
            self.assertEqual([item['type'] for item in prompt],
                             ['image', 'text'])
            self.assertEqual(prompt[0]['value'], 'c.png')
            self.assertEqual('Think step by step' in prompt[1]['value'],
                             dataset_name.endswith('_COT'))

    def test_opencompass_and_vlmeval_image_encoding_match(self):
        from PIL import Image
        from vlmeval.smp import encode_image_file_to_base64

        from opencompass.models import OpenAISDK

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / 'image.png'
            Image.new('RGB', (32, 24), 'blue').save(image_path)
            model = object.__new__(OpenAISDK)
            model.image_format = 'JPEG'
            model.image_min_edge = 100
            encoded = model._chat_image_url(str(image_path)).split(',', 1)[1]
            native = encode_image_file_to_base64(str(image_path))

        self.assertEqual(encoded, native)

    def test_opencompass_and_vlmeval_request_messages_match(self):
        from PIL import Image
        from vlmeval.api import LMDeployAPI

        from opencompass.models import OpenAISDK

        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / 'first.png'
            second = Path(temp_dir) / 'second.png'
            Image.new('RGB', (32, 24), 'blue').save(first)
            Image.new('RGB', (200, 150), 'red').save(second)
            prompt = [
                dict(type='text', value='before'),
                dict(type='image', value=str(first)),
                dict(type='text', value='middle'),
                dict(type='image', value=str(second)),
                dict(type='text', value='after'),
            ]
            native_model = object.__new__(LMDeployAPI)
            native_model.local_media = False
            native = native_model.prepare_inputs(prompt, None)
            oc_model = object.__new__(OpenAISDK)
            oc_model.image_format = 'JPEG'
            oc_model.image_min_edge = 100
            opencompass = oc_model._messages_to_chat_completions(
                convert_vlmeval_prompt(prompt))

        self.assertEqual(opencompass, native)

    def _assert_evaluator_parity(self, dataset_name, predictions):
        from vlmeval.smp.status_report import flatten_summary_metrics

        native_dataset = self._dataset(dataset_name)
        frame = native_dataset.data.drop(columns='image').copy()
        frame['prediction'] = predictions
        with tempfile.TemporaryDirectory() as temp_dir:
            native_dir = Path(temp_dir) / 'native'
            oc_dir = Path(temp_dir) / 'opencompass'
            native_dir.mkdir()
            native_file = native_dir / f'{dataset_name}.xlsx'
            frame.to_excel(native_file, index=False)
            native_metrics = native_dataset.evaluate(str(native_file),
                                                     model='exact_matching',
                                                     nproc=1)

            evaluator = VLMEvalKitEvaluator(dataset_name=dataset_name,
                                            eval_kwargs=dict(
                                                model='exact_matching',
                                                nproc=1))
            evaluator._out_dir = str(oc_dir)
            sample_ids = [
                f'{dataset_name}:{index}'
                for index in native_dataset.data['index']
            ]
            test_set = Dataset.from_list([
                dict(sample_id=sample_id,
                     vlmeval_index=index,
                     dataset_name=dataset_name)
                for sample_id, index in zip(
                    sample_ids, native_dataset.data['index'].tolist())
            ])
            with patch(
                    'opencompass.evaluator.vlmevalkit.'
                    'build_vlmeval_dataset',
                    return_value=self._dataset(dataset_name)):
                oc_metrics = evaluator.score(predictions=predictions,
                                             test_set=test_set)

            oc_file = oc_dir / f'{dataset_name}.xlsx'
            pd.testing.assert_frame_equal(pd.read_excel(native_file),
                                          pd.read_excel(oc_file))
            pd.testing.assert_frame_equal(
                pd.read_csv(
                    native_file.with_name(f'{native_file.stem}_acc.csv')),
                pd.read_csv(oc_file.with_name(f'{oc_file.stem}_acc.csv')))
            flattened = flatten_summary_metrics(native_metrics)
            for key, value in flattened.items():
                self.assertAlmostEqual(oc_metrics[key], float(value) * 100)
            raw_metrics = json.loads(
                (oc_dir /
                 'vlmevalkit_metrics.json').read_text(encoding='utf-8'))
            self.assertEqual(raw_metrics['flattened'], flattened)

    def test_mmbench_same_predictions_match_native_evaluator(self):
        self._assert_evaluator_parity('MMBench_DEV_EN', ['A', 'A'])

    def test_mmmu_pro_same_predictions_match_native_evaluator(self):
        self._assert_evaluator_parity('MMMU_Pro_10c', ['B', 'B'])

    def test_mmmu_pro_cot_same_predictions_match_native_evaluator(self):
        self._assert_evaluator_parity(
            'MMMU_Pro_10c_COT',
            ['reasoning\nAnswer: $B', 'reasoning\nAnswer: B'])


class TestVLMEvalKitEndToEnd(unittest.TestCase):

    def test_existing_opencompass_inference_to_vlmeval_evaluate(self):
        fake = FakeVLMEvalDataset()
        model = FakeModel()
        with tempfile.TemporaryDirectory() as temp_dir, patch(
                'opencompass.datasets.vlmevalkit.dataset.'
                'build_vlmeval_dataset',
                return_value=fake):
            dataset = VLMEvalKitDataset(abbr='MMBench_DEV_EN',
                                        dataset_name='MMBench_DEV_EN',
                                        reader_cfg=dict(
                                            input_columns=['prompt'],
                                            output_column='answer'))
            retriever = ZeroRetriever(dataset)
            template = RawPromptTemplate(
                messages=[dict(expand_column='prompt')],
                format_variables=False)
            inferencer = GenInferencer(model=model,
                                       max_out_len=16,
                                       batch_size=2,
                                       output_json_filepath=temp_dir,
                                       output_json_filename='predictions.json')
            predictions = inferencer.inference(retriever,
                                               prompt_template=template)
            artifact = json.loads(
                (Path(temp_dir) /
                 'predictions.json').read_text(encoding='utf-8'))

            self.assertEqual(predictions, ['A', 'B'])
            self.assertNotIn('sample_id', artifact['0'])
            self.assertEqual(model.inputs[0][0]['content'][0]['type'], 'image')
            self.assertEqual(model.inputs[0][0]['content'][0]['image_url'],
                             '/images/101.jpg')

            def to_excel(frame, path, index=False):
                Path(path).touch()

            records = list(artifact.values())
            task = object.__new__(OpenICLEvalTask)
            task.eval_cfg = dict(evaluator=dict(type=VLMEvalKitEvaluator,
                                                dataset_name='MMBench_DEV_EN',
                                                eval_kwargs=dict(
                                                    model='exact_matching')))
            task.model_cfg = dict(abbr='fake-model')
            task.dataset_cfg = dict(abbr='MMBench_DEV_EN', k=1, n=1)
            task.work_dir = temp_dir
            task.output_column = 'answer'
            task.dump_details = False
            task.cal_extract_rate = False
            with patch(
                    'opencompass.evaluator.vlmevalkit.build_vlmeval_dataset',
                    return_value=fake), patch(
                        'opencompass.evaluator.vlmevalkit.'
                        '_flatten_vlmeval_metrics',
                        side_effect=fake_flatten_metrics), patch.object(
                            pd.DataFrame, 'to_excel', new=to_excel):
                metrics = task._evaluate_predictions(
                    [record['prediction'] for record in records], dataset.test,
                    records)

            self.assertEqual(metrics['split=dev|Overall'], 50.0)
            self.assertNotIn('artifacts', metrics)
            self.assertEqual(len(fake.evaluate_calls), 1)

    def test_mmmu_pro_opencompass_inference_to_vlmeval_evaluate(self):
        fake = FakeMMMUProDataset()
        model = FakeModel()
        with tempfile.TemporaryDirectory() as temp_dir, patch(
                'opencompass.datasets.vlmevalkit.dataset.'
                'build_vlmeval_dataset',
                return_value=fake):
            dataset = VLMEvalKitDataset(abbr='MMMU_Pro_10c',
                                        dataset_name='MMMU_Pro_10c',
                                        reader_cfg=dict(
                                            input_columns=['prompt'],
                                            output_column='answer'))
            retriever = ZeroRetriever(dataset)
            template = RawPromptTemplate(
                messages=[dict(expand_column='prompt')],
                format_variables=False)
            inferencer = GenInferencer(model=model,
                                       max_out_len=16,
                                       batch_size=2,
                                       output_json_filepath=temp_dir,
                                       output_json_filename='predictions.json')
            predictions = inferencer.inference(retriever,
                                               prompt_template=template)
            artifact = json.loads(
                (Path(temp_dir) /
                 'predictions.json').read_text(encoding='utf-8'))

            self.assertEqual(predictions, ['B', 'B'])
            self.assertNotIn('sample_id', artifact['0'])
            self.assertEqual(
                [part['type'] for part in model.inputs[0][0]['content']],
                ['text', 'image', 'text', 'image', 'text'])

            records = list(artifact.values())
            task = object.__new__(OpenICLEvalTask)
            task.eval_cfg = dict(evaluator=dict(type=VLMEvalKitEvaluator,
                                                dataset_name='MMMU_Pro_10c',
                                                eval_kwargs=dict(
                                                    model='exact_matching')))
            task.model_cfg = dict(abbr='fake-model')
            task.dataset_cfg = dict(abbr='MMMU_Pro_10c', k=1, n=1)
            task.work_dir = temp_dir
            task.output_column = 'answer'
            task.dump_details = False
            task.cal_extract_rate = False

            def to_excel(frame, path, index=False):
                Path(path).touch()

            with patch(
                    'opencompass.evaluator.vlmevalkit.'
                    'build_vlmeval_dataset',
                    return_value=fake), patch(
                        'opencompass.evaluator.vlmevalkit.'
                        '_flatten_vlmeval_metrics',
                        side_effect=fake_flatten_metrics), patch.object(
                            pd.DataFrame, 'to_excel', new=to_excel):
                metrics = task._evaluate_predictions(
                    [record['prediction'] for record in records], dataset.test,
                    records)

            self.assertEqual(metrics['Overall Acc'], 50.0)
            self.assertEqual(metrics['split=none|History'], 100.0)
            self.assertEqual(len(fake.evaluate_calls), 1)


if __name__ == '__main__':
    unittest.main()
