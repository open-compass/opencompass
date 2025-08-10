from opencompass.models import HuggingFace

# refer to https://github.com/THUDM/CodeGeeX2/tree/main
# For pass@1   : n=20 , temperature=0.2, top_p=0.95
# For Pass@10  : n=200, temperature=0.8, top_p=0.95
# For Pass@100 : n=200, temperature=0.8, top_p=0.95

models = [
    dict(
        type=HuggingFace,
        abbr='codegeex2-6b',
        path='THUDM/codegeex2-6b',
        tokenizer_path='THUDM/codegeex2-6b',
        tokenizer_kwargs=dict(
           padding_side='left',
           truncation_side='left',
           trust_remote_code=True,
        ),
        max_out_len=1024,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(trust_remote_code=True, device_map='auto'),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
