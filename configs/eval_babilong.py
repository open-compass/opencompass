from mmengine.config import read_base

with read_base():
    # Models
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_5_7b_chat import (
        models as lmdeploy_internlm2_5_7b_chat_model,
    )
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_7b_instruct import (
        models as lmdeploy_qwen2_5_7b_instruct_model,
    )
    from opencompass.configs.models.hf_llama.lmdeploy_llama3_1_8b_instruct import (
        models as lmdeploy_llama3_1_8b_instruct_model,
    )
    from opencompass.configs.models.mistral.lmdeploy_ministral_8b_instruct_2410 import (
        models as lmdeploy_ministral_8b_instruct_2410_model,
    )

    # Datasets
    from opencompass.configs.datasets.babilong.babilong_0k_gen import (
        babiLong_0k_datasets,
    )
    from opencompass.configs.datasets.babilong.babilong_4k_gen import (
        babiLong_4k_datasets,
    )
    from opencompass.configs.datasets.babilong.babilong_16k_gen import (
        babiLong_16k_datasets,
    )
    from opencompass.configs.datasets.babilong.babilong_32k_gen import (
        babiLong_32k_datasets,
    )
    from opencompass.configs.datasets.babilong.babilong_128k_gen import (
        babiLong_128k_datasets,
    )
    from opencompass.configs.datasets.babilong.babilong_256k_gen import (
        babiLong_256k_datasets,
    )
    from opencompass.configs.summarizers.groups.babilong import (
        babilong_summary_groups,
    )

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

models = sum([v for k, v in locals().items() if k.endswith('_model')], [])
for model in models:
    model['engine_config']['session_len'] = 1024 * 1024
    model['max_seq_len'] = 1024 * 1024
    model['engine_config']['tp'] = 4
    model['run_cfg']['num_gpus'] = 4


summarizer = dict(
    dataset_abbrs=[
        'babilong_0k',
        'babilong_4k',
        'babilong_16k',
        'babilong_32k',
        'babilong_128k',
        'babilong_256k',
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []
    ),
)

work_dir = './outputs/babilong'
