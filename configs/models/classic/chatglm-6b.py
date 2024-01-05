from opencompass.models import HuggingFace

models = [
    # chatglm
    dict(abbr='chatglm-6b',
        type=HuggingFace, path='THUDM/chatglm-6b',
        model_kwargs=dict(trust_remote_code=True), tokenizer_kwargs=dict(trust_remote_code=True),
        max_out_len=100, max_seq_len=2048, batch_size=16, run_cfg=dict(num_gpus=1)),
]
