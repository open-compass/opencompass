from opencompass.models import TurboMindModelwithChatTemplate
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='qwen_3_0.6b_thinking-turbomind',
        path='/cpfs01/shared/llm_ddd/opencompass/models/hf_hub/models--Qwen--Qwen3-0.6B/snapshots/ec0dcdfc641f7a94f8e969d459caa772b9c01706',
        engine_config=dict(session_len=32768, max_batch_size=16, tp=1),
        gen_config=dict(
            top_k=20, temperature=0.6, top_p=0.95, do_sample=True, enable_thinking=True
        ),
        max_seq_len=32768,
        max_out_len=32000,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    ),
]