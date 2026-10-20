from copy import deepcopy

from mmengine.config import read_base

from opencompass.models import GeminiSDK, OpenAISDK, OpenAISDKResponse
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

with read_base():
    from opencompass.configs.datasets.InverseIFEval.InverseIFEval_rawprompt_gen import (  # noqa: E501
        inverse_ifeval_datasets,
    )
    from opencompass.configs.summarizers.inverse_ifeval import summarizer  # noqa: F401, E501


_o4_mini_reasoning_high = dict(effort='high', summary=None)
_gemini_2_5_thinking = dict(thinking_budget=1024, include_thoughts=True)

models = [
    dict(
        abbr='GPT-4.1',
        type=OpenAISDK,
        path='gpt-4.1',
        key='ENV',
        query_per_second=1,
        max_out_len=4096,
        max_seq_len=1047576,
        batch_size=8,
        temperature=0.0,
        retry=10),
]

# Official judge model matrix from the Hugging Face dataset card.
_inverse_ifeval_judge_model_by_type = {
    'QC': 'o4-mini.high',
    'ITF': 'Gemini-2.5-Pro',
    'CC': 'Gemini-2.5-Flash',
    'CCF': 'o4-mini.high',
    'DIA': 'Gemini-2.5-Flash',
    'II': 'DeepSeek-V3-0324',
    'MIM': 'Gemini-2.0-flash.001',
    'CA': 'Gemini-2.5-Pro',
}


def _build_o4_mini_high_judge_cfg():
    return dict(
        type=OpenAISDKResponse,
        path='o4-mini',
        key='ENV',
        query_per_second=1,
        batch_size=8,
        max_out_len=164000,
        max_seq_len=1047576,
        openai_extra_kwargs=dict(reasoning=_o4_mini_reasoning_high),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
        retry=10,
    )


def _build_gemini_judge_cfg(path, thinking=None):
    cfg = dict(
        type=GeminiSDK,
        path=path,
        query_per_second=1,
        batch_size=8,
        temperature=0.0,
        max_out_len=4096,
        max_seq_len=49152,
        pred_postprocessor=dict(type=extract_non_reasoning_content),
        retry=10,
    )
    if thinking is not None:
        cfg['thinking'] = thinking
    return cfg


def _build_deepseek_v3_0324_judge_cfg():
    return dict(
        type=OpenAISDK,
        path='deepseek-v3-250324',
        key='ENV',
        query_per_second=1,
        batch_size=8,
        temperature=0.0,
        tokenizer_path='gpt-4o-2024-05-13',
        max_out_len=4096,
        max_seq_len=49152,
        retry=10,
    )


def _build_inverse_ifeval_judge_cfg(instruction_type):
    judge_model = _inverse_ifeval_judge_model_by_type[instruction_type]
    if judge_model == 'o4-mini.high':
        return _build_o4_mini_high_judge_cfg()
    if judge_model == 'Gemini-2.5-Pro':
        return _build_gemini_judge_cfg('gemini-2.5-pro',
                                       _gemini_2_5_thinking)
    if judge_model == 'Gemini-2.5-Flash':
        return _build_gemini_judge_cfg('gemini-2.5-flash',
                                       _gemini_2_5_thinking)
    if judge_model == 'DeepSeek-V3-0324':
        return _build_deepseek_v3_0324_judge_cfg()
    if judge_model == 'Gemini-2.0-flash.001':
        return _build_gemini_judge_cfg('gemini-2.0-flash-001')
    raise ValueError(f'Unsupported InverseIFEval judge model: {judge_model}')


datasets = []
for dataset in inverse_ifeval_datasets:
    dataset = deepcopy(dataset)
    instruction_type = dataset['abbr'].rsplit('_', 1)[-1]
    dataset['eval_cfg']['evaluator']['judge_cfg'] = (
        _build_inverse_ifeval_judge_cfg(instruction_type))
    datasets.append(dataset)

del _build_inverse_ifeval_judge_cfg
del _build_o4_mini_high_judge_cfg
del _build_gemini_judge_cfg
del _build_deepseek_v3_0324_judge_cfg
del _inverse_ifeval_judge_model_by_type
del _o4_mini_reasoning_high
del _gemini_2_5_thinking
del GeminiSDK
del OpenAISDK
del OpenAISDKResponse
del extract_non_reasoning_content
del deepcopy
del read_base
del inverse_ifeval_datasets
del dataset
del instruction_type
