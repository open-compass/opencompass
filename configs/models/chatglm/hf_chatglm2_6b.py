from opencompass.models import HuggingFace


models = [
    dict(
        type=HuggingFace,
        abbr='chatglm2-6b-hf',
        path='THUDM/chatglm2-6b',
        tokenizer_path='THUDM/chatglm2-6b',
        tokenizer_kwargs=dict(
           padding_side='left',
           truncation_side='left',
           trust_remote_code=True,
        ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(trust_remote_code=True, device_map='auto', revision='a6d54fac46dff2db65d53416c207a4485ca6bd40'),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
