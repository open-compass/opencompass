import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock

from opencompass.openicl.icl_inferencer.icl_agent_inferencer import (
    AgentInferencer, AgentInferencerOutputHandler)
from opencompass.openicl.icl_inferencer.icl_base_inferencer import \
    ChatOutputHandler
from opencompass.openicl.icl_inferencer.icl_chat_inferencer import \
    ChatInferencer


class TestAgentInferencer(unittest.TestCase):
    modes = ('last', 'every', 'every_with_gt')

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.chats = [[
            message for turn in range(2) for message in (
                dict(role='user', content=f'问题{sample}-{turn}，中文\u0085\u2028'),
                dict(role='assistant', content=f'参考{sample}-{turn}'))
        ] for sample in range(2)]
        self.retriever = MagicMock()
        self.retriever.retrieve.side_effect = lambda: deepcopy([[], []])

    @staticmethod
    def _reply(prompt, phase):
        answer = phase + '回答：' + prompt
        steps = [dict(type='工具', result=phase + '轨迹：' + prompt)]
        return answer, steps, [dict(role='assistant', content=answer)]

    def _model(self, phase=''):
        model = MagicMock()
        model.template_parser = object()
        model.is_api = True
        model.agent._llm._generate_is_wrapped = False
        model.agent._llm.generate.return_value = ['适配结果']
        model.chat.side_effect = (
            lambda prompt, *args: self._reply(prompt, phase))
        model.gt_response.side_effect = lambda gold: [
            dict(role='assistant', content=gold)
        ]
        model.generate_from_template.side_effect = lambda prompts, **kwargs: [
            phase + '回答：' + prompts[0][-1]['content']
        ]
        return model

    def _inferencer(self, cls, model, directory, mode):
        inferencer = cls(model=model,
                         output_json_filepath=str(directory),
                         output_json_filename='results.json',
                         infer_mode=mode,
                         save_every=1,
                         max_out_len=16)
        inferencer.is_main_process = True
        inferencer.get_chat_list = MagicMock(
            side_effect=lambda *args, **kwargs: deepcopy(self.chats))
        return inferencer

    def _prompts(self, mode, samples=range(2)):
        turns = (2, ) if mode == 'last' else (0, 2)
        return [
            self.chats[sample][turn]['content'] for sample in samples
            for turn in turns
        ]

    def _expected(self, mode, sample, phase='', agent=True):
        chat = self.chats[sample]
        turns = (2, ) if mode == 'last' else (0, 2)
        results = []
        for turn in turns:
            prompt = chat[turn]['content']
            answer, steps, _ = self._reply(prompt, phase)
            result = dict(
                gold=chat[turn + 1]['content'],
                prediction=answer,
                origin_prompt=prompt if agent else deepcopy(chat[:turn + 1]))
            if agent:
                result['steps'] = steps
            results.append(result)
        if mode == 'last':
            return results[0]
        return {key: [item[key] for item in results] for key in results[0]}

    @staticmethod
    def _read_jsonl(path):
        with path.open(encoding='utf-8') as stream:
            return [json.loads(line) for line in stream]

    def _assert_finished(self, directory, filename, results):
        self.assertEqual({path.name
                          for path in directory.iterdir()}, {filename})
        saved = json.loads((directory / filename).read_text(encoding='utf-8'))
        self.assertEqual(saved, results)

    def test_handler_empty_checkpoint(self):
        handler = AgentInferencerOutputHandler()
        self.assertIsInstance(handler, ChatOutputHandler)
        path = self.directory / 'empty.jsonl'
        self.assertEqual(handler.restore_from_jsonl(self.directory, path.name),
                         {})
        self.assertFalse(path.exists())
        path.touch()
        restored = AgentInferencerOutputHandler()
        self.assertEqual(
            restored.restore_from_jsonl(self.directory, path.name), {})
        self.assertEqual(restored.dumped_indices, set())

    def test_handler_roundtrip_and_append(self):
        for multiround in (False, True):
            with self.subTest(multiround=multiround):
                filename = f'checkpoint-{multiround}.jsonl'
                path = self.directory / filename
                handler = AgentInferencerOutputHandler()
                expected = self._expected('every' if multiround else 'last', 0)
                save = (handler.save_multiround_results
                        if multiround else handler.save_results)
                for turn in ((0, 2) if multiround else (2, )):
                    prompt = self.chats[0][turn]['content']
                    answer, steps, _ = self._reply(prompt, '')
                    save(prompt,
                         answer,
                         steps,
                         0,
                         gold=self.chats[0][turn + 1]['content'])
                self.assertEqual(handler.results_dict, {'0': expected})
                handler.write_to_jsonl(self.directory, filename)
                original = path.read_bytes()
                self.assertIn('中文', original.decode('utf-8'))
                handler.write_to_jsonl(self.directory, filename)
                self.assertEqual(path.read_bytes(), original)
                restored = AgentInferencerOutputHandler()
                self.assertEqual(
                    restored.restore_from_jsonl(self.directory, filename),
                    {'0': expected})
                self.assertEqual(restored.dumped_indices, {'0'})
                restored.write_to_jsonl(self.directory, filename)
                self.assertEqual(path.read_bytes(), original)
                second = self._expected('last', 1)
                restored.save_results(second['origin_prompt'],
                                      second['prediction'],
                                      second['steps'],
                                      1,
                                      gold=second['gold'])
                restored.write_to_jsonl(self.directory, filename)
                restored.write_to_jsonl(self.directory, filename)
                self.assertEqual(restored.dumped_indices, {'0', '1'})
                self.assertEqual(
                    [row['idx'] for row in self._read_jsonl(path)], ['0', '1'])
                final = AgentInferencerOutputHandler()
                self.assertEqual(
                    final.restore_from_jsonl(self.directory, filename), {
                        '0': expected,
                        '1': second
                    })
                self.assertEqual(final.dumped_indices, {'0', '1'})
                final.write_to_json(self.directory, 'handler.json')
                self.assertEqual(
                    json.loads((self.directory /
                                'handler.json').read_text(encoding='utf-8')),
                    final.results_dict)

    def test_agent_cold_start_all_modes(self):
        for mode in self.modes:
            with self.subTest(mode=mode):
                directory = self.directory / mode
                model = self._model()
                inferencer = self._inferencer(AgentInferencer, model,
                                              directory, mode)
                self.assertEqual(model.agent._llm.generate('探测'), '适配结果')
                results = inferencer.inference(self.retriever)
                self.assertEqual(
                    results, {
                        str(sample): self._expected(mode, sample)
                        for sample in range(2)
                    })
                self.assertEqual(
                    [call.args[0] for call in model.chat.call_args_list],
                    self._prompts(mode))
                self.assertEqual(model.reset.call_count, 2)
                if mode == 'every_with_gt':
                    self.assertEqual([
                        call.args[0]
                        for call in model.gt_response.call_args_list
                    ], [chat[1]['content'] for chat in self.chats])
                self.retriever.retrieve.assert_called_once_with()
                inferencer.get_chat_list.assert_called_once_with(
                    [[], []], self.retriever, prompt_template=None)
                self.retriever.reset_mock()
                self._assert_finished(directory, 'results.json', results)

    def _check_resume(self, cls, mode, interrupt_turn=0):
        agent = cls is AgentInferencer
        directory = self.directory / f'{cls.__name__}-{mode}-{interrupt_turn}'
        filename = 'renamed.output.json'
        checkpoint = directory / 'tmp_renamed.output.jsonl'
        model = self._model('before-')
        method = model.chat if agent else model.generate_from_template
        reply = method.side_effect
        interrupted_prompt = self._prompts(mode, [1])[interrupt_turn]

        def interrupt(*args, **kwargs):
            prompt = args[0] if agent else args[0][0][-1]['content']
            if prompt == interrupted_prompt:
                raise RuntimeError('模拟第二个样本中断')
            return reply(*args, **kwargs)

        method.side_effect = interrupt
        inferencer = self._inferencer(cls, model, directory, mode)
        with self.assertRaisesRegex(RuntimeError, '模拟第二个样本中断'):
            inferencer.inference(self.retriever, output_json_filename=filename)
        self.assertEqual({path.name
                          for path in directory.iterdir()}, {checkpoint.name})
        rows = self._read_jsonl(checkpoint)
        self.assertEqual([row['idx'] for row in rows], ['0'])
        old = rows[0].copy()
        old.pop('idx')
        self.assertEqual(old, self._expected(mode, 0, 'before-', agent))

        fresh_model = self._model('after-')
        resumed = self._inferencer(cls, fresh_model, directory, mode)
        results = resumed.inference(self.retriever,
                                    output_json_filename=filename)
        self.assertEqual(results, {
            '0': old,
            '1': self._expected(mode, 1, 'after-', agent)
        })
        if agent:
            calls = fresh_model.chat.call_args_list
            actual = [call.args[0] for call in calls]
            self.assertEqual(results['0']['steps'], old['steps'])
            self.assertEqual(fresh_model.reset.call_count, 1)
        else:
            calls = fresh_model.generate_from_template.call_args_list
            actual = [call.args[0][0][-1]['content'] for call in calls]
            self.assertEqual(calls[0].args[0], [self.chats[1][:-1]])
            self.assertEqual(calls[0].kwargs, dict(max_out_len=16))
            fresh_model.chat.assert_not_called()
        self.assertEqual(actual, self._prompts(mode, [1]))
        self.assertFalse(checkpoint.exists())
        self._assert_finished(directory, filename, results)

    def test_agent_resume_all_modes(self):
        for mode in self.modes:
            with self.subTest(mode=mode):
                self._check_resume(AgentInferencer, mode)

    def test_agent_resume_restarts_incomplete_multiround_sample(self):
        for mode in ('every', 'every_with_gt'):
            with self.subTest(mode=mode):
                self._check_resume(AgentInferencer, mode, interrupt_turn=1)

    def test_chat_resume_last(self):
        self._check_resume(ChatInferencer, 'last')


if __name__ == '__main__':
    unittest.main()
