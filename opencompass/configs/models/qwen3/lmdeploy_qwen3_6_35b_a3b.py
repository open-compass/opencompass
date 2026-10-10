from opencompass.models import TurboMindModelwithChatTemplate
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='qwen3.6-35b-a3b-lmdeploy',
        path='Qwen/Qwen3.6-35B-A3B',
        engine_config=dict(session_len=262144, max_batch_size=8, tp=2),
        gen_config=dict(
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            do_sample=True,
        ),
        max_seq_len=262144,
        max_out_len=131072,
        batch_size=8,
        pred_postprocessor=dict(type=extract_non_reasoning_content),
        run_cfg=dict(num_gpus=2),
        max_workers=8,
    )
]
