from opencompass.models import TurboMindModelwithChatTemplate

models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='deepseek_r1_distill_llama_70b_turbomind',
        path='deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
        engine_config=dict(max_batch_size=16, tp=4),
        gen_config=dict(top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=8192),
        max_seq_len=16384,
        max_out_len=8192,
        batch_size=16,
        run_cfg=dict(num_gpus=4),
        stop_words=['<|end_of_text|>', '<|eot_id|>', '<|eom_id|>'],
    )
]
