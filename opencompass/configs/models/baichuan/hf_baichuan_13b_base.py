from opencompass.models import HuggingFaceCausalLM


models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='baichuan-13b-base-hf',
        path='baichuan-inc/Baichuan-13B-Base',
        tokenizer_path='baichuan-inc/Baichuan-13B-Base',
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                              use_fast=False,),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True, revision='77d74f449c4b2882eac9d061b5a0c4b7c1936898'),
        run_cfg=dict(num_gpus=2, num_procs=1),
    )
]
