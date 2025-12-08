from mmengine.config import read_base

from opencompass.models import (HuggingFacewithChatTemplate,
                                TurboMindModelwithChatTemplate,
                                VLLMwithChatTemplate)
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

with read_base():
    # choose a list of datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import \
        gsm8k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.race.race_gen import \
        race_datasets  # noqa: F401, E501

    # re-design .. including some models and modify all kinds of configs
    from ...rjob import eval, infer  # noqa: F401, E501

Qwen3_0_6B_FP8_hf = dict(
    type=HuggingFacewithChatTemplate,
    abbr='qwen3_0_6b_fp8-hf',
    path='Qwen/Qwen3-0.6B-FP8',
    max_out_len=16384,
    batch_size=8,
    run_cfg=dict(num_gpus=1),
    pred_postprocessor=dict(type=extract_non_reasoning_content))

Qwen3_0_6B_FP8_turbomind = dict(
    type=TurboMindModelwithChatTemplate,
    abbr='qwen3-0_6b-fp8-turbomind',
    path='Qwen/Qwen3-0.6B-FP8',
    engine_config=dict(session_len=32768, max_batch_size=1),
    gen_config=dict(top_k=1, max_new_tokens=16384),
    max_seq_len=32768,
    max_out_len=16384,
    batch_size=1,
    run_cfg=dict(num_gpus=1),
    pred_postprocessor=dict(type=extract_non_reasoning_content))

Qwen3_0_6B_FP8_vllm = dict(
    type=VLLMwithChatTemplate,
    abbr='qwen3-0_6b-fp8-vllm',
    path='Qwen/Qwen3-0.6B-FP8',
    model_kwargs=dict(tensor_parallel_size=1),
    generation_kwargs=dict(temperature=0),  # greedy
    max_seq_len=32768,
    max_out_len=16384,
    batch_size=1,
    run_cfg=dict(num_gpus=1),
)

race_datasets = [race_datasets[1]]
datasets = sum([v for k, v in locals().items() if k.endswith('_datasets')], [])

for d in datasets:
    d['reader_cfg']['test_range'] = '[0:32]'

models = sum([v for k, v in locals().items() if k.endswith('_model')], [])

models = [Qwen3_0_6B_FP8_hf, Qwen3_0_6B_FP8_turbomind, Qwen3_0_6B_FP8_vllm]

summarizer = dict(
    dataset_abbrs=[
        'gsm8k',
        'race-middle',
        'race-high',
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
