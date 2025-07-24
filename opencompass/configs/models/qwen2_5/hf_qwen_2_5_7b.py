from opencompass.models import HuggingFaceBaseModel

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='qwen2.5-7b-hf',
        path='/cpfs01/shared/llm_ddd/opencompass/models/hf_hub/models--Qwen--Qwen2.5-7B/snapshots/09a0bac5707b43ec44508eab308b0846320c1ed4',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    )
]