from opencompass.models import TurboMindModelwithChatTemplate
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='deepseek-r1-distill-llama-70b-turbomind',
        path='deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
        engine_config=dict(session_len=32768, max_batch_size=8, tp=8),
        gen_config=dict(top_k=1,
                        temperature=1e-6,
                        top_p=0.9,
                        max_new_tokens=32768),
        max_seq_len=32768,
        max_out_len=32768,
        batch_size=8,
        run_cfg=dict(num_gpus=8),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
