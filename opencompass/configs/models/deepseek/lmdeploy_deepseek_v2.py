# flake8: noqa
from mmengine.config import read_base
from opencompass.models import (
    TurboMindModel,
)
lmdeploy_deepseek_v2_model = [
    dict(
        type=TurboMindModel,
        abbr='deepseek-v2-turbomind',
        path='deepseek-ai/DeepSeek-V2',
        engine_config=dict(session_len=7168, max_batch_size=4, tp=8, cache_max_entry_count=0.7),
        gen_config=dict(top_k=1, temperature=1e-6, top_p=0.9),
        max_seq_len=7168,
        max_out_len=2048,
        batch_size=4,
        run_cfg=dict(num_gpus=8),
    )
]