from opencompass.models import HuggingFaceCausalLM


models = [dict(
        type=HuggingFaceCausalLM,
        abbr='autoj-13b',
        path='GAIR/autoj-13b',
        tokenizer_path='GAIR/autoj-13b',
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                              use_fast=False,),
        max_out_len=1024,
        max_seq_len=4096,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )]
