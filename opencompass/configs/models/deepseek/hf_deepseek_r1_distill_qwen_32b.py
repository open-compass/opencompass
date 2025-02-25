from opencompass.models import HuggingFacewithChatTemplate
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='deepseek-r1-distill-qwen-32b-hf',
        path='deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
        max_out_len=16384,
        batch_size=8,
        run_cfg=dict(num_gpus=4),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
