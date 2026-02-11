from opencompass.models import TurboMindModel, TurboMindModelwithChatTemplate
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

models = [
    dict(type=TurboMindModelwithChatTemplate,
         abbr='qwen-3-8b-fullbench',
         path='Qwen/Qwen3-8B',
         engine_config=dict(session_len=32768, max_batch_size=1, tp=1),
         gen_config=dict(do_sample=False, enable_thinking=True),
         max_seq_len=32768,
         max_out_len=32768,
         batch_size=1,
         run_cfg=dict(num_gpus=1),
         pred_postprocessor=dict(type=extract_non_reasoning_content))
]

interns1_models = [
    dict(type=TurboMindModelwithChatTemplate,
         abbr='intern-s1-pro',
         path='intern/Intern-S1-Pro',
         engine_config=dict(session_len=32768, max_batch_size=1, tp=16),
         gen_config=dict(do_sample=False, enable_thinking=True),
         max_seq_len=32768,
         max_out_len=32768,
         batch_size=1,
         run_cfg=dict(num_gpus=16),
         pred_postprocessor=dict(type=extract_non_reasoning_content))
]

judge_models = [
    dict(type=TurboMindModelwithChatTemplate,
         abbr='qwen-3-8b-fullbench',
         path='Qwen/Qwen3-8B',
         engine_config=dict(session_len=46000, max_batch_size=1, tp=1),
         gen_config=dict(do_sample=False, enable_thinking=True),
         max_seq_len=46000,
         max_out_len=46000,
         batch_size=1,
         run_cfg=dict(num_gpus=1),
         pred_postprocessor=dict(type=extract_non_reasoning_content))
]

base_models = [
    dict(
        type=TurboMindModel,
        abbr='qwen-3-8b-base-fullbench',
        path='Qwen/Qwen3-8B-Base',
        engine_config=dict(session_len=32768, max_batch_size=1, tp=1),
        gen_config=dict(top_k=1,
                        temperature=1e-6,
                        top_p=0.9,
                        max_new_tokens=1024),
        max_seq_len=32768,
        max_out_len=1024,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]
