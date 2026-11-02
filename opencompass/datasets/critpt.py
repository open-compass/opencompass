import copy
import errno
import json
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from datasets import Dataset, DownloadConfig, load_dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.registry import ICL_INFERENCERS, LOAD_DATASET

from .base import BaseDataset

DEFAULT_HF_DATASET = 'CritPt-Benchmark/CritPt'
_CRITPT_PARSE_PROMPT_KEY = '_critpt_parse_prompt'
_CRITPT_STEP_INDEX_KEY = '_critpt_step_index'
_CRITPT_STEP_NAME_KEY = '_critpt_step_name'
_CRITPT_PROBLEM_ID_KEY = '_critpt_problem_id'
_CRITPT_PART_KEY = '_critpt_part'
_CRITPT_DATASET_NAME_KEY = '_critpt_dataset_name'
_CRITPT_SOURCE_PATH_KEY = '_critpt_source_path'

_DEFAULT_SYSTEM_PROMPT_TWO_STEP = (
    'You are a physics research assistant specializing in solving complex, '
    'research-level problems using precise, step-by-step reasoning.\n\n'
    '**Input**\n'
    'Problems will be provided in Markdown format.\n\n'
    '**Output (Markdown format)**\n\n'
    '1. **Step-by-Step Derivation** - Show every non-trivial step in the '
    'solution.  Justify steps using relevant physical laws, theorems, or '
    'mathematical identities.\n'
    '2. **Mathematical Typesetting** - Use LaTeX for all mathematics:  '
    '`$...$` for inline expressions, `$$...$$` for display equations.\n'
    '3. **Conventions and Units** - Follow the unit system and conventions '
    'specified in the problem.\n'
    '4. **Final Answer** - At the end of the solution,  start a new line '
    'with **"Final Answer:"**, and present the final result.\n\n'
    '    For final answers involving values, follow the precision '
    'requirements specified in the problem.\n'
    '    If no precision is specified:\n'
    '    - If an exact value is possible, provide it (e.g., '
    '\\$\\sqrt(2)\\$, \\$\\pi/4\\$).\n'
    '    - If exact form is not feasible, retain at least 12 significant '
    'digits in the result. \n\n'
    '5. **Formatting Compliance** - If the user requests a specific output '
    'format (e.g., code, table),  provide the final answer accordingly.')

_DEFAULT_SYSTEM_PROMPT_ONE_STEP = (
    'You are a physics research assistant specializing in solving complex, '
    'research-level problems using precise, step-by-step reasoning.\n\n'
    '**Input**\n'
    'Problems will be provided in Markdown format.\n\n'
    '**Output (Markdown format)**\n\n'
    '1. **Step-by-Step Derivation** - Show every non-trivial step in the '
    'solution.  Justify steps using relevant physical laws, theorems, or '
    'mathematical identities.\n'
    '2. **Mathematical Typesetting** - Use LaTeX for all mathematics:  '
    '`$...$` for inline expressions, `$$...$$` for display equations.\n'
    '3. **Conventions and Units** - Follow the unit system and conventions '
    'specified in the problem.\n'
    '4. **Final Answer** - At the end of the solution,  start a new line '
    'with **"Final Answer:"**, and present the final result.\n\n'
    '    For final answers involving values, follow the precision '
    'requirements specified in the problem.\n'
    '    If no precision is specified:\n'
    '    - If an exact value is possible, provide it (e.g., '
    '\\$\\sqrt(2)\\$, \\$\\pi/4\\$).\n'
    '    - If exact form is not feasible, retain at least 12 significant '
    'digits in the result. \n\n'
    '5. **Parsing Structure** - After obtaining the final answer,  populate '
    'it into the code template provided at the end of the problem. This step '
    'is purely for formatting/display purposes. No additional reasoning or '
    'derivation should be performed. Do not import any modules or packages '
    'beyond what is provided in the template. ')


def _build_default_system_prompt(parsing: bool,
                                 use_python: bool = False,
                                 use_web_search: bool = False) -> str:
    system_prompt = (_DEFAULT_SYSTEM_PROMPT_ONE_STEP
                     if parsing else _DEFAULT_SYSTEM_PROMPT_TWO_STEP)

    if use_python:
        needle = ('Justify steps using relevant physical laws, theorems, or '
                  'mathematical identities.')
        replacement = ('Justify steps using relevant physical laws, theorems, '
                       'mathematical identities or numerical codes.')
        system_prompt = system_prompt.replace(needle, replacement)

    if use_web_search:
        reminder = (
            '\nYou must use web search engine to gather all the necessary '
            'information before solving the problem.')
        system_prompt = f'{system_prompt}{reminder}{reminder}{reminder}'

    return system_prompt


def _build_default_parse_prompt(code_template: str) -> str:
    code_template = _content_to_text(code_template)
    return (
        'Populate your final answer into the code template provided below. '
        'This step is purely for formatting/display purposes. No additional '
        'reasoning or derivation should be performed. Do not import any '
        'modules or packages beyond what is provided in the template.\n'
        '```python\n'
        f'{code_template}\n'
        '```')


@dataclass
class _StepSpec:
    name: str
    problem_id: str
    prompt: str
    code_template: str
    target: str


def _load_hf_split(path: str) -> Dataset:
    try:
        return load_dataset(path, split='train')
    except OSError as exc:
        if getattr(exc, 'errno', None) not in (errno.EACCES, errno.EROFS):
            raise
        return _load_hf_split_with_tmp_cache(path)


def _load_hf_split_with_tmp_cache(path: str) -> Dataset:
    default_cache_root = Path.cwd() / 'tmp' / 'opencompass_hf'
    cache_root = Path(
        os.path.expanduser(os.environ.get('HF_HOME', str(default_cache_root))))
    datasets_cache = Path(
        os.path.expanduser(
            os.environ.get('HF_DATASETS_CACHE', str(cache_root / 'datasets'))))
    hub_cache = Path(
        os.path.expanduser(
            os.environ.get(
                'HF_HUB_CACHE',
                os.environ.get('HUGGINGFACE_HUB_CACHE',
                               str(cache_root / 'hub')))))
    xet_cache = Path(
        os.path.expanduser(
            os.environ.get('HF_XET_CACHE', str(cache_root / 'xet'))))

    import huggingface_hub.constants as hf_constants

    env_keys = (
        'HF_HOME',
        'HF_DATASETS_CACHE',
        'HF_HUB_CACHE',
        'HUGGINGFACE_HUB_CACHE',
        'HF_XET_CACHE',
        'HF_HUB_DISABLE_XET',
    )
    old_env = {key: os.environ.get(key) for key in env_keys}
    constant_keys = (
        'HF_HOME',
        'HF_HUB_CACHE',
        'HUGGINGFACE_HUB_CACHE',
        'HF_XET_CACHE',
        'HF_HUB_DISABLE_XET',
    )
    old_constants = {
        key: getattr(hf_constants, key, None)
        for key in constant_keys
    }

    try:
        os.environ['HF_HOME'] = str(cache_root)
        os.environ['HF_DATASETS_CACHE'] = str(datasets_cache)
        os.environ['HF_HUB_CACHE'] = str(hub_cache)
        os.environ['HUGGINGFACE_HUB_CACHE'] = str(hub_cache)
        os.environ['HF_XET_CACHE'] = str(xet_cache)
        os.environ['HF_HUB_DISABLE_XET'] = '1'
        hf_constants.HF_HOME = str(cache_root)
        hf_constants.HF_HUB_CACHE = str(hub_cache)
        hf_constants.HUGGINGFACE_HUB_CACHE = str(hub_cache)
        hf_constants.HF_XET_CACHE = str(xet_cache)
        hf_constants.HF_HUB_DISABLE_XET = True

        download_config = DownloadConfig(cache_dir=str(hub_cache))
        return load_dataset(path,
                            split='train',
                            cache_dir=str(datasets_cache),
                            download_config=download_config)
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        for key, value in old_constants.items():
            setattr(hf_constants, key, value)


def _hf_dataset_name(row: Dict[str, Any]) -> str:
    source_path = _content_to_text(row.get('metadata_notebook_path', ''))
    if source_path:
        dataset_name = Path(source_path).stem
    else:
        problem_id = _content_to_text(row.get('problem_id', ''))
        dataset_name = (problem_id[:-5]
                        if problem_id.endswith('_main') else problem_id)
    dataset_name = dataset_name.replace(' ', '_').replace('-', '_')
    while '__' in dataset_name:
        dataset_name = dataset_name.replace('__', '_')
    return dataset_name


def _hf_main_step(row: Dict[str, Any], parsing: bool) -> _StepSpec:
    dataset_name = _hf_dataset_name(row)
    problem_id = _content_to_text(row.get('problem_id', '')).strip()
    if not problem_id:
        problem_id = f'{dataset_name}_main'
    prompt = _content_to_text(row.get('problem_description', ''))
    code_template = _content_to_text(row.get('code_template', '') or '')
    if parsing and code_template:
        fence = chr(96) * 3
        prompt = f'{prompt}\n\n{fence}python\n{code_template}\n{fence}'
    return _StepSpec(
        name='main',
        problem_id=problem_id,
        prompt=prompt,
        code_template=code_template,
        target=_content_to_text(row.get('answer_only_code', '') or ''),
    )


def _message(role: str, content: str) -> Dict[str, str]:
    return {'role': role, 'content': content}


def _generation_slot(dataset_name: str, source_path: str, step: _StepSpec,
                     parse_prompt: str, step_index: int,
                     part: str) -> Dict[str, Any]:
    message = _message('assistant', '')
    message.update({
        _CRITPT_PARSE_PROMPT_KEY: parse_prompt,
        _CRITPT_STEP_INDEX_KEY: step_index,
        _CRITPT_STEP_NAME_KEY: step.name,
        _CRITPT_PROBLEM_ID_KEY: step.problem_id,
        _CRITPT_PART_KEY: part,
        _CRITPT_DATASET_NAME_KEY: dataset_name,
        _CRITPT_SOURCE_PATH_KEY: source_path,
    })
    return message


@LOAD_DATASET.register_module()
class CritPtDataset(BaseDataset):
    """OpenCompass dataset for CritPt public HuggingFace data."""

    @staticmethod
    def load(path: str = DEFAULT_HF_DATASET,
             parsing: bool = False,
             use_python: bool = False,
             use_web_search: bool = False,
             debug_num_samples: Optional[int] = None) -> Dataset:
        hf_path = DEFAULT_HF_DATASET if path == 'CritPt' else path
        hf_dataset = _load_hf_split(hf_path)
        if debug_num_samples is not None:
            hf_dataset = hf_dataset.select(
                range(min(debug_num_samples, len(hf_dataset))))

        rows = []
        system_prompt = _build_default_system_prompt(parsing, use_python,
                                                     use_web_search)
        for row in hf_dataset:
            step = _hf_main_step(row, parsing)
            dataset_name = _hf_dataset_name(row)
            source_path = _content_to_text(
                row.get('metadata_notebook_path', ''))
            rows.append(
                _hf_step_row(dataset_name, source_path, step, system_prompt,
                             parsing, use_python, use_web_search))
        return Dataset.from_list(rows)


def _hf_step_row(dataset_name: str, source_path: str, step: _StepSpec,
                 system_prompt: str, parsing: bool, use_python: bool,
                 use_web_search: bool) -> Dict[str, Any]:
    parse_prompt = '' if parsing else _build_default_parse_prompt(
        step.code_template)
    return {
        'dataset_name':
        dataset_name,
        'source_path':
        source_path,
        'part':
        'main',
        'problem_id':
        step.problem_id,
        'problem_ids': [step.problem_id],
        'step_names': [step.name],
        'system_prompt':
        system_prompt,
        'messages': [
            _message('system', system_prompt),
            _message('user', step.prompt),
            _generation_slot(dataset_name, source_path, step, parse_prompt, 0,
                             'main'),
        ],
        'step_prompts': [step.prompt],
        'parse_prompts': [parse_prompt],
        'code_templates': [step.code_template],
        'target':
        step.target,
        'targets': [step.target],
        'parsing':
        parsing,
        'use_golden_for_prev_steps':
        False,
        'multiturn_with_answer':
        False,
        'use_python':
        use_python,
        'use_web_search':
        use_web_search,
    }


class CritPtEvaluator(BaseEvaluator):
    """Prepare official CritPt submissions and optionally submit them."""

    DEFAULT_SERVER_URL = 'https://artificialanalysis.ai/api/v2/critpt/evaluate'
    DEFAULT_API_KEY_ENVS = (
        'CRITPT_API_KEY',
        'ARTIFICIAL_ANALYSIS_API_KEY',
        'AA_API_KEY',
    )

    def __init__(self,
                 submit: bool = False,
                 submit_env: str = 'CRITPT_SUBMIT_TO_API',
                 server_url: str = DEFAULT_SERVER_URL,
                 api_key: Optional[str] = None,
                 api_key_env: Optional[Union[str, List[str],
                                             Tuple[str, ...]]] = None,
                 model: Optional[str] = None,
                 generation_config: Optional[Dict[str, Any]] = None,
                 timeout: float = 7200.0,
                 output_subdir: str = 'critpt_submission',
                 save_individual: bool = True,
                 include_messages: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.submit = submit
        self.submit_env = submit_env
        self.server_url = server_url
        self.api_key = api_key
        self.api_key_env = api_key_env or self.DEFAULT_API_KEY_ENVS
        assert model is not None, ('CritPtEvaluator 需要传一个和评测模型对应的 model。')
        self.model = model
        self.generation_config = generation_config or {}
        self.timeout = timeout
        self.output_subdir = output_subdir
        self.save_individual = save_individual
        self.include_messages = include_messages

    def score(self, predictions, references, test_set):
        submissions = self._build_submissions(predictions, test_set)
        output_dir = Path(getattr(self, '_out_dir',
                                  './critpt_eval')) / self.output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.save_individual:
            indiv_dir = output_dir / 'submissions'
            indiv_dir.mkdir(parents=True, exist_ok=True)
            for submission in submissions:
                self._dump_json(indiv_dir / f"{submission['problem_id']}.json",
                                submission)

        batch_metadata = {
            'source': 'opencompass',
            'total_submissions': len(submissions),
            'timestamp': datetime.now().isoformat(),
        }
        payload = {
            'submissions':
            [self._wire_submission(submission) for submission in submissions],
            'batch_metadata':
            batch_metadata,
        }
        payload_path = output_dir / 'batch_submission.json'
        self._dump_json(payload_path, payload)

        result = {
            'generated': len(predictions),
            'submission_count': len(submissions),
            'submission_path': str(payload_path),
            'submitted': 0,
        }

        if self._should_submit():
            server_result = self._submit_payload(payload)
            server_result_path = output_dir / 'aggregate_report.json'
            self._dump_json(
                server_result_path, {
                    'timestamp':
                    datetime.now().isoformat(),
                    'summary':
                    self._server_summary(server_result, len(submissions)),
                    'metrics':
                    server_result,
                })
            result['submitted'] = 1
            result['server_result_path'] = str(server_result_path)
            result.update(self._numeric_metrics(server_result))

        return result

    def _build_submissions(self, predictions,
                           test_set) -> List[Dict[str, Any]]:
        if len(predictions) != len(test_set):
            raise ValueError('CritPt predictions and test_set have different '
                             f'lengths: {len(predictions)} vs {len(test_set)}')

        submissions = []
        seen_problem_ids = set()
        for sample_idx, (prediction,
                         sample) in enumerate(zip(predictions, test_set)):
            problem_ids = sample.get('problem_ids') or [sample['problem_id']]
            prediction_list = (prediction if isinstance(prediction, list) else
                               [prediction])
            if len(problem_ids) != len(prediction_list):
                raise ValueError(
                    'CritPt prediction count does not match problem_ids for '
                    f'sample {sample_idx}: {len(prediction_list)} vs '
                    f'{len(problem_ids)}')

            for problem_id, generated_code in zip(problem_ids,
                                                  prediction_list):
                if problem_id in seen_problem_ids:
                    raise ValueError(f'Duplicate CritPt problem_id: '
                                     f'{problem_id}')
                seen_problem_ids.add(problem_id)
                submissions.append({
                    'problem_id':
                    problem_id,
                    'generated_code':
                    _content_to_text(generated_code),
                    'model':
                    self.model,
                    'timestamp':
                    datetime.now().isoformat(),
                    'generation_config':
                    self._generation_config(sample),
                    'messages':
                    self._submission_messages(sample, generated_code),
                })
        return submissions

    def _generation_config(self, sample) -> Dict[str, Any]:
        config = copy.deepcopy(self.generation_config)
        config.update({
            'use_golden_for_prev_steps':
            sample.get('use_golden_for_prev_steps'),
            'parsing':
            sample.get('parsing'),
            'multiturn_with_answer':
            sample.get('multiturn_with_answer'),
            'use_python':
            sample.get('use_python'),
            'use_web_search':
            sample.get('use_web_search'),
        })
        return config

    def _submission_messages(self, sample,
                             generated_code) -> Optional[List[Dict[str, str]]]:
        if not self.include_messages:
            return None
        return [_message('assistant', _content_to_text(generated_code))]

    @staticmethod
    def _wire_submission(submission: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'problem_id': submission['problem_id'],
            'generated_code': submission['generated_code'],
            'model': submission['model'],
            'generation_config': submission['generation_config'],
            'messages': submission['messages'],
        }

    def _should_submit(self) -> bool:
        return self.submit or _str_to_bool(os.environ.get(self.submit_env))

    def _resolve_api_key(self) -> Optional[str]:
        if self.api_key:
            return self.api_key
        envs = ([self.api_key_env]
                if isinstance(self.api_key_env, str) else self.api_key_env)
        for env in envs:
            value = os.environ.get(env)
            if value:
                return value
        return None

    def _submit_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        import requests

        api_key = self._resolve_api_key()
        if not api_key:
            raise ValueError(
                'CritPt official submission requested, but no API key was '
                'provided. Set api_key or one of api_key_env variables.')

        headers = {'x-api-key': api_key}
        response = requests.post(self.server_url,
                                 json=payload,
                                 headers=headers,
                                 timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _server_summary(server_result: Dict[str, Any],
                        total_submissions: int) -> Dict[str, Any]:
        return {
            'total_submissions': total_submissions,
            'accuracy': server_result.get('accuracy', 0.0),
            'timeout_rate': server_result.get('timeout_rate', 0.0),
            'server_timeout_count': server_result.get('server_timeout_count',
                                                      0),
        }

    @staticmethod
    def _numeric_metrics(server_result: Dict[str, Any]) -> Dict[str, Any]:
        metrics = {}
        for key, value in server_result.items():
            if isinstance(value, (int, float)):
                metrics[key] = value
        return metrics

    @staticmethod
    def _dump_json(path: Path, data: Dict[str, Any]) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def _str_to_bool(value: Optional[str]) -> bool:
    return str(value).lower() in {'1', 'true', 'yes', 'y', 'on'}


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get('text', item.get('content', item))
                parts.append(str(text))
            elif hasattr(item, 'text'):
                parts.append(str(item.text))
            elif hasattr(item, 'reasoning'):
                parts.append(str(item.reasoning))
            else:
                parts.append(str(item))
        return ''.join(parts)
    return str(content)


def _normalize_messages(
        messages: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [{
        'role': str(message['role']),
        'content': _content_to_text(message.get('content', '')),
    } for message in messages]


@ICL_INFERENCERS.register_module()
class CritPtInferencer(GenInferencer):
    """CritPt multiround inferencer with an extra parse call per reasoning."""

    def __init__(self,
                 *args,
                 dump_infer_message_path: Optional[str] = None,
                 dump_infer_messages_only: bool = False,
                 **kwargs):
        kwargs['multiround'] = True
        super().__init__(*args, **kwargs)
        self.dump_infer_message_path = (
            dump_infer_message_path
            or os.environ.get('CRITPT_DUMP_INFER_MESSAGE_PATH'))
        self.dump_infer_messages_only = dump_infer_messages_only
        self._dump_file_initialized = False
        self._dump_sample_index = 0

    def _generate_multiround(
            self, entry: List[List[Dict[str, Any]]],
            extra_gen_kwargs: dict) -> List[Union[str, List[str]]]:
        """Fill reasoning slots with the base implementation, then parse them.

        CritPt's public data only has main problems, but keeping the parse call
        based on a truncated history also preserves the official sub-chain
        semantics for the example challenge: parse outputs are not inherited by
        later reasoning turns.
        """
        slot_indices = [self._reasoning_slot_indices(chat) for chat in entry]
        clean_entry = [[
            _message(str(message['role']),
                     _content_to_text(message.get('content', '')))
            for message in chat
        ] for chat in entry]

        if not self.dump_infer_messages_only:
            super()._generate_multiround(clean_entry, extra_gen_kwargs)

        parse_gen_indices = []
        predictions_by_chat = []
        for chat_idx, indices in enumerate(slot_indices):
            sample_index = self._dump_sample_index + chat_idx
            chat_predictions = []
            gen_indices = []
            for step_order, message_idx in enumerate(indices):
                reasoning = _content_to_text(
                    clean_entry[chat_idx][message_idx].get('content', ''))
                entry[chat_idx][message_idx]['content'] = reasoning

                reasoning_messages = _normalize_messages(
                    clean_entry[chat_idx][:message_idx])
                slot = entry[chat_idx][message_idx]
                self._dump_messages(reasoning_messages, slot, sample_index,
                                    step_order, 'reasoning')

                parse_prompt = slot.get(_CRITPT_PARSE_PROMPT_KEY, '')
                if not parse_prompt:
                    chat_predictions.append(reasoning)
                    continue

                gen_indices.append(
                    (step_order, message_idx, len(chat_predictions)))
                chat_predictions.append('')
            parse_gen_indices.append(gen_indices)
            predictions_by_chat.append(chat_predictions)

        def _gen_parse_turn(chat_idx, step):
            step_order, message_idx, pred_idx = parse_gen_indices[chat_idx][
                step]
            slot = entry[chat_idx][message_idx]
            parse_messages = _normalize_messages(
                entry[chat_idx][:message_idx + 1] +
                [_message('user', slot[_CRITPT_PARSE_PROMPT_KEY])])
            sample_index = self._dump_sample_index + chat_idx
            self._dump_messages(parse_messages, slot, sample_index, step_order,
                                'parse')
            if self.dump_infer_messages_only:
                return
            output = self.model.generate_from_template(
                [parse_messages],
                max_out_len=self.max_out_len,
                **extra_gen_kwargs,
            )[0]
            predictions_by_chat[chat_idx][pred_idx] = _content_to_text(output)

        next_step = [0] * len(entry)
        in_flight = {}

        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            for chat_idx in range(len(entry)):
                if parse_gen_indices[chat_idx]:
                    in_flight[executor.submit(_gen_parse_turn, chat_idx,
                                              0)] = chat_idx
                    next_step[chat_idx] = 1

            while in_flight:
                done, _ = wait(set(in_flight), return_when=FIRST_COMPLETED)
                for future in done:
                    chat_idx = in_flight.pop(future)
                    future.result()
                    ns = next_step[chat_idx]
                    if ns < len(parse_gen_indices[chat_idx]):
                        in_flight[executor.submit(_gen_parse_turn, chat_idx,
                                                  ns)] = chat_idx
                        next_step[chat_idx] = ns + 1

        predictions = [
            chat_predictions[0]
            if len(chat_predictions) == 1 else chat_predictions
            for chat_predictions in predictions_by_chat
        ]

        self._dump_sample_index += len(entry)
        return predictions

    @staticmethod
    def _reasoning_slot_indices(chat: List[Dict[str, Any]]) -> List[int]:
        return [
            i for i, message in enumerate(chat)
            if (isinstance(message, dict) and message.get('role') ==
                'assistant' and not message.get('content', ''))
        ]

    def _dump_messages(self, messages: List[Dict[str, str]], slot: Dict[str,
                                                                        Any],
                       index: int, step_order: int, call_type: str) -> None:
        if not self.dump_infer_message_path or not self.is_main_process:
            return
        dump_path = Path(self.dump_infer_message_path)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        mode = 'a' if self._dump_file_initialized else 'w'
        record = {
            'index': index,
            'dataset_name': slot.get(_CRITPT_DATASET_NAME_KEY),
            'source_path': slot.get(_CRITPT_SOURCE_PATH_KEY),
            'part': slot.get(_CRITPT_PART_KEY),
            'step_index': slot.get(_CRITPT_STEP_INDEX_KEY, step_order),
            'step_name': slot.get(_CRITPT_STEP_NAME_KEY),
            'problem_id': slot.get(_CRITPT_PROBLEM_ID_KEY),
            'call_type': call_type,
            'messages': messages,
        }
        with open(dump_path, mode, encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        self._dump_file_initialized = True
