from mmengine.config import read_base

from opencompass.models import VLLM, HuggingFaceBaseModel, TurboMindModel

with read_base():
    # choose a list of datasets
    from opencompass.configs.datasets.gpqa.gpqa_openai_simple_evals_gen_5aeece import \
        gpqa_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_17d0dc import \
        gsm8k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.race.race_ppl import \
        race_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.winogrande.winogrande_5shot_ll_252f01 import \
        winogrande_datasets  # noqa: F401, E501

    from ...rjob import eval, infer  # noqa: F401, E501

race_datasets = [race_datasets[1]]
models = [
    dict(
        type=TurboMindModel,
        abbr='qwen3-8b-base-turbomind',
        path='Qwen/Qwen3-0.6B-Base',
        engine_config=dict(max_batch_size=1, tp=1),
        gen_config=dict(top_k=1,
                        temperature=1e-6,
                        top_p=0.9,
                        max_new_tokens=2048),
        max_seq_len=8192,
        max_out_len=2048,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    ),
    dict(
        type=VLLM,
        abbr='qwen3-8b-base-vllm',
        path='Qwen/Qwen3-0.6B-Base',
        model_kwargs=dict(tensor_parallel_size=1, gpu_memory_utilization=0.6),
        max_seq_len=8192,
        max_out_len=2048,
        batch_size=16,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=1),
    ),
    dict(type=HuggingFaceBaseModel,
         abbr='qwen3-8b-base-hf',
         path='Qwen/Qwen3-0.6B-Base',
         max_out_len=1024,
         batch_size=4,
         run_cfg=dict(num_gpus=1))
]
datasets = sum([v for k, v in locals().items() if k.endswith('_datasets')], [])

for d in datasets:
    d['reader_cfg']['test_range'] = '[0:32]'

for m in models:
    if 'turbomind' in m['abbr'] or 'lmdeploy' in m['abbr']:
        m['engine_config']['max_batch_size'] = 1
        m['batch_size'] = 1
models = sorted(models, key=lambda x: x['run_cfg']['num_gpus'])

summarizer = dict(
    dataset_abbrs=[
        ['gsm8k', 'accuracy'],
        ['GPQA_diamond', 'accuracy'],
        ['race-high', 'accuracy'],
        ['winogrande', 'accuracy'],
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
