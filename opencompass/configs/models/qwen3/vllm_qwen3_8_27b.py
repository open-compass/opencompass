from opencompass.models import VLLMwithChatTemplate
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='qwen3.8-27b-vllm',
        path='Qwen/Qwen3.8-27B',
        model_kwargs=dict(
            tensor_parallel_size=2,
            max_model_len=262144,
            trust_remote_code=True,
        ),
        chat_template_kwargs=dict(enable_thinking=True),
        generation_kwargs=dict(temperature=0.6, top_p=0.95, top_k=20),
        max_seq_len=262144,
        max_out_len=131072,
        batch_size=8,
        pred_postprocessor=dict(type=extract_non_reasoning_content),
        run_cfg=dict(num_gpus=2),
        max_workers=8,
    )
]
