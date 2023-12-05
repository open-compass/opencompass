from opencompass.models import HuggingFace


models = [
    dict(
        type=HuggingFace,
        abbr='chatglm3-6b-base-hf',
        path='THUDM/chatglm3-6b-base',
        tokenizer_path='THUDM/chatglm3-6b-base',
        model_kwargs=dict(
            trust_remote_code=True,
            device_map='auto',
        ),
        tokenizer_kwargs=dict(
           padding_side='left',
           truncation_side='left',
           trust_remote_code=True,
        ),
        max_out_len=100,
        max_seq_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
