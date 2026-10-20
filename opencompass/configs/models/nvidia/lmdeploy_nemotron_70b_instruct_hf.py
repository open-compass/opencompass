from opencompass.models import TurboMindModelwithChatTemplate

models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='nvidia-3_1-Nemotron-70b-instruct-HF-turbomind',
        path='nvidia/Llama-3.1-Nemotron-70B-Instruct-HF',
        engine_config=dict(max_batch_size=16, tp=4),
        gen_config=dict(
            top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=4096
        ),
        max_seq_len=16384,
        max_out_len=4096,
        batch_size=16,
        run_cfg=dict(num_gpus=4),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
    )
]
