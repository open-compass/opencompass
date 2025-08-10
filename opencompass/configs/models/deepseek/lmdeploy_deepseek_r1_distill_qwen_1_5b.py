from opencompass.models import TurboMindModelwithChatTemplate
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='deepseek-r1-distill-qwen-1_5b-turbomind',
        path='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
        engine_config=dict(session_len=32768, max_batch_size=16, tp=1),
        gen_config=dict(top_k=1,
                        temperature=1e-6,
                        top_p=0.9,
                        max_new_tokens=32768),
        max_seq_len=32768,
        max_out_len=32768,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
