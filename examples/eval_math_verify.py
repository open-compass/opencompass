from mmengine.config import read_base
from opencompass.models import TurboMindModelwithChatTemplate
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

with read_base():
    from opencompass.configs.datasets.math.math_500_gen import math_datasets

models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='deepseek-r1-distill-llama-8b-turbomind',
        path='deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
        engine_config=dict(session_len=32768, max_batch_size=8, tp=1),
        gen_config=dict(
            top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=4096
        ),
        max_seq_len=32768,
        max_out_len=32768,
        batch_size=32,
        run_cfg=dict(num_gpus=1),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    ),
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='deepseek-r1-distill-qwen-7b-turbomind',
        path='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
        engine_config=dict(session_len=32768, max_batch_size=8, tp=1),
        gen_config=dict(
            temperature=0.6,
            top_p=0.95,
            max_new_tokens=32768,
            do_sample=True,
        ),
        max_seq_len=32768,
        max_out_len=32768,
        batch_size=32,
        run_cfg=dict(num_gpus=1),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    ),
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='deepseek-r1-distill-qwen-1_5b-turbomind',
        path='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
        engine_config=dict(session_len=32768, max_batch_size=16, tp=1),
        gen_config=dict(
            top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=4096
        ),
        max_seq_len=32768,
        max_out_len=32768,
        batch_size=32,
        run_cfg=dict(num_gpus=1),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    ),
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='deepseek-r1-distill-qwen-14b-turbomind',
        path='deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',
        engine_config=dict(session_len=32768, max_batch_size=16, tp=2),
        gen_config=dict(
            top_k=1,
            temperature=0.6,
            top_p=0.95,
            max_new_tokens=32768,
            do_sample=True,
        ),
        max_seq_len=32768,
        max_out_len=32768,
        batch_size=16,
        run_cfg=dict(num_gpus=2),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    ),
]

datasets = [*math_datasets]


work_dir = './outputs/math_500'
