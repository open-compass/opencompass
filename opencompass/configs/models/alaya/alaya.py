from opencompass.models import AlayaLM


models = [
    dict(
        type=AlayaLM,
        abbr='alaya-7b-hf',
        path='DataCanvas/Alaya-7B-Base',
        tokenizer_path='DataCanvas/Alaya-7B-Base',
        tokenizer_kwargs=dict(padding_side='left',
                        truncation_side='left',
                        trust_remote_code=True,
                        use_fast=False,),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1))
]
