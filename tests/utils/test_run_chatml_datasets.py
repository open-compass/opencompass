"""Unit tests for ChatML dataset config construction."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
CHATOBJ_CUSTOM_GEN = (
    'opencompass.configs.datasets.chatobj_custom.chatobj_custom_gen')


def _make_package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _make_module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _load_chatobj_custom_gen():
    module_path = REPO_ROOT / 'opencompass' / 'configs' / 'datasets' \
        / 'chatobj_custom' / 'chatobj_custom_gen.py'
    spec = importlib.util.spec_from_file_location(CHATOBJ_CUSTOM_GEN,
                                                  module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_logger():

    class Logger:

        def debug(self, *args, **kwargs):
            pass

        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

    return Logger()


class Config(dict):

    @classmethod
    def fromfile(cls, *args, **kwargs):
        return cls()

    def merge_from_dict(self, cfg):
        self.update(cfg)


def _make_run_stubs():
    model_names = [
        'VLLM',
        'HuggingFace',
        'HuggingFaceBaseModel',
        'HuggingFaceCausalLM',
        'HuggingFaceChatGLM3',
        'HuggingFacewithChatTemplate',
        'TurboMindModelwithChatTemplate',
        'VLLMwithChatTemplate',
    ]
    task_names = ['OpenICLEvalTask', 'OpenICLInferTask']
    runner_names = ['DLCRunner', 'LocalRunner', 'SlurmRunner']
    partitioner_names = ['NaivePartitioner', 'NumWorkerPartitioner']

    chatobj_custom_gen = _load_chatobj_custom_gen()

    return {
        'mmengine':
        _make_package('mmengine'),
        'mmengine.config':
        _make_module('mmengine.config', Config=Config),
        'tabulate':
        _make_module('tabulate',
                     tabulate=lambda *args, **kwargs: 'formatted table'),
        'opencompass':
        _make_package('opencompass'),
        'opencompass.datasets':
        _make_package('opencompass.datasets'),
        'opencompass.datasets.custom':
        _make_module(
            'opencompass.datasets.custom',
            make_custom_dataset_config=lambda dataset: dataset,
        ),
        'opencompass.models':
        _make_module('opencompass.models',
                     **{name: type(name, (), {})
                        for name in model_names}),
        'opencompass.partitioners':
        _make_module(
            'opencompass.partitioners',
            **{name: type(name, (), {})
               for name in partitioner_names},
        ),
        'opencompass.runners':
        _make_module('opencompass.runners',
                     **{name: type(name, (), {})
                        for name in runner_names}),
        'opencompass.tasks':
        _make_module('opencompass.tasks',
                     **{name: type(name, (), {})
                        for name in task_names}),
        'opencompass.utils':
        _make_module(
            'opencompass.utils',
            get_logger=_make_logger,
            match_files=lambda *args, **kwargs: [],
        ),
        'opencompass.configs':
        _make_package('opencompass.configs'),
        'opencompass.configs.datasets':
        _make_package('opencompass.configs.datasets'),
        'opencompass.configs.datasets.chatobj_custom':
        _make_package('opencompass.configs.datasets.chatobj_custom'),
        CHATOBJ_CUSTOM_GEN:
        chatobj_custom_gen,
    }


class TestConstructChatMLDatasets(unittest.TestCase):

    def setUp(self):
        self.stub_modules = _make_run_stubs()
        self.module_patch = patch.dict(sys.modules, self.stub_modules)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

        chatobj_custom_gen = self.stub_modules[CHATOBJ_CUSTOM_GEN]
        self.default_answer_pattern = chatobj_custom_gen.optional_evaluator[
            'mcq_rule_evaluator']['pred_postprocessor']['answer_pattern']
        self.default_grader_template = chatobj_custom_gen.optional_evaluator[
            'llm_evaluator']['prompt_template']['messages'][1]['content']

        run_path = REPO_ROOT / 'opencompass' / 'utils' / 'run.py'
        spec = importlib.util.spec_from_file_location(
            'opencompass_utils_run_under_test', run_path)
        self.run_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.run_module)

    def test_mcq_rule_evaluator_uses_available_template(self):
        datasets = self.run_module.consturct_chatml_datasets([
            dict(
                abbr='custom_mcq',
                path='mcq.jsonl',
                evaluator=dict(
                    type='mcq_rule_evaluator',
                    answer_pattern=r'Answer:\s*([A-D])',
                ),
            )
        ])

        evaluator = datasets[0]['eval_cfg']['evaluator']
        self.assertEqual(evaluator['type'], 'AccEvaluator')
        self.assertEqual(evaluator['pred_postprocessor']['answer_pattern'],
                         r'Answer:\s*([A-D])')
        self.assertEqual(datasets[0]['type'], 'ChatMLDataset')

    def test_mcq_rule_evaluator_overrides_do_not_leak(self):
        datasets = self.run_module.consturct_chatml_datasets([
            dict(
                abbr='custom_pattern',
                path='custom.jsonl',
                evaluator=dict(
                    type='mcq_rule_evaluator',
                    answer_pattern=r'Choice:\s*([A-D])',
                ),
            ),
            dict(
                abbr='default_pattern',
                path='default.jsonl',
                evaluator=dict(type='mcq_rule_evaluator'),
            ),
        ])

        first = datasets[0]['eval_cfg']['evaluator']['pred_postprocessor']
        second = datasets[1]['eval_cfg']['evaluator']['pred_postprocessor']
        self.assertEqual(first['answer_pattern'], r'Choice:\s*([A-D])')
        self.assertEqual(second['answer_pattern'], self.default_answer_pattern)
        self.assertIsNot(first, second)

    def test_llm_evaluator_prompt_overrides_do_not_leak(self):
        datasets = self.run_module.consturct_chatml_datasets([
            dict(
                abbr='custom_prompt',
                path='custom.jsonl',
                evaluator=dict(
                    type='llm_evaluator',
                    judge_cfg=dict(abbr='judge-a'),
                    prompt='custom grading prompt',
                ),
            ),
            dict(
                abbr='default_prompt',
                path='default.jsonl',
                evaluator=dict(
                    type='llm_evaluator',
                    judge_cfg=dict(abbr='judge-b'),
                ),
            ),
        ])

        first_messages = datasets[0]['eval_cfg']['evaluator'][
            'prompt_template']['messages']
        second_messages = datasets[1]['eval_cfg']['evaluator'][
            'prompt_template']['messages']
        self.assertEqual(first_messages[1]['content'], 'custom grading prompt')
        self.assertEqual(second_messages[1]['content'],
                         self.default_grader_template)
        self.assertIsNot(first_messages, second_messages)

    def test_reader_cfg_overrides_do_not_leak(self):
        datasets = self.run_module.consturct_chatml_datasets([
            dict(
                abbr='ranged',
                path='ranged.jsonl',
                test_range='[0:1]',
                evaluator=dict(type='math_evaluator'),
            ),
            dict(
                abbr='unranged',
                path='unranged.jsonl',
                evaluator=dict(type='math_evaluator'),
            ),
        ])

        self.assertEqual(datasets[0]['reader_cfg']['test_range'], '[0:1]')
        self.assertNotIn('test_range', datasets[1]['reader_cfg'])

    def test_change_accelerator_vllm_preserves_generation_kwargs(self):
        model = dict(
            type=self.run_module.HuggingFacewithChatTemplate,
            abbr='chat-hf',
            path='test/model',
            max_seq_len=4096,
            max_out_len=1024,
            model_kwargs=dict(dtype='auto'),
            generation_kwargs=dict(
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                max_new_tokens=1024,
                min_new_tokens=8,
                eos_token_id=[1, 2],
            ),
            batch_size=8,
            run_cfg=dict(num_gpus=2),
            stop_words=['<stop>'],
            min_out_len=4,
            summarizer_abbr='chat-summary',
        )

        converted = self.run_module.change_accelerator([model], 'vllm')[0]

        self.assertEqual(converted['abbr'], 'chat-vllm')
        self.assertEqual(converted['model_kwargs'], {
            'tensor_parallel_size': 2,
            'max_model_len': 4096,
            'dtype': 'auto',
        })
        self.assertEqual(
            converted['generation_kwargs'], {
                'temperature': 0.6,
                'top_p': 0.95,
                'min_tokens': 8,
                'stop_token_ids': [1, 2],
            })
        self.assertEqual(converted['stop_words'], ['<stop>'])
        self.assertEqual(converted['min_out_len'], 4)
        self.assertEqual(converted['summarizer_abbr'], 'chat-summary')

    def test_change_accelerator_vllm_accepts_gen_config_alias(self):
        model = dict(
            type=self.run_module.HuggingFacewithChatTemplate,
            abbr='deepseek-r1-distill-qwen-1.5b-hf',
            path='hf_models/DeepSeek-R1-Distill-Qwen-1.5B',
            max_seq_len=33792,
            max_out_len=32768,
            gen_config=dict(
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                max_new_tokens=32768,
            ),
            batch_size=64,
            run_cfg=dict(num_gpus=1),
            pred_postprocessor=dict(type='extract_non_reasoning_content'),
        )

        converted = self.run_module.change_accelerator([model], 'vllm')[0]

        self.assertEqual(converted['abbr'],
                         'deepseek-r1-distill-qwen-1.5b-vllm')
        self.assertEqual(converted['model_kwargs'], {
            'tensor_parallel_size': 1,
            'max_model_len': 33792,
        })
        self.assertEqual(converted['generation_kwargs'], {
            'temperature': 0.6,
            'top_p': 0.95,
        })
        self.assertEqual(converted['pred_postprocessor'],
                         dict(type='extract_non_reasoning_content'))


if __name__ == '__main__':
    unittest.main()
