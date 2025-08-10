from opencompass.models import HuggingFacewithChatTemplate
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='huatuogpt-o1-8b-hf',
        path='FreedomIntelligence/HuatuoGPT-o1-8B',
        max_out_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1),
        pred_postprocessor=dict(type=extract_non_reasoning_content, think_start_token='## Thinking', think_end_token='## Final Response'),
    )
]
