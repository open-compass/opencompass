from opencompass.models import VLLM


models = [
    dict(
        type=VLLM,
        abbr='mistral-7b-v0.2-vllm',
        path='mistral-community/Mistral-7B-v0.2',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=32,
        model_kwargs=dict(dtype='bfloat16'),
        generation_kwargs=dict(temperature=0, top_p=1, max_tokens=2048, stop_token_ids=[2]),
        run_cfg=dict(num_gpus=1, num_procs=1),
        stop_words=['[INST]'],
    )
]
