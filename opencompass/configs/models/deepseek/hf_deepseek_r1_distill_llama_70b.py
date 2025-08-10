from opencompass.models import HuggingFacewithChatTemplate
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='deepseek-r1-distill-llama-70b-hf',
        path='deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
        max_out_len=16384,
        batch_size=8,
        run_cfg=dict(num_gpus=8),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
