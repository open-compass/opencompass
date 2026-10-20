from opencompass.models import HuggingFaceCausalLM


models = [dict(
        type=HuggingFaceCausalLM,
        abbr='judgelm-33b-v1-hf',
        path='BAAI/JudgeLM-33B-v1.0',
        tokenizer_path='BAAI/JudgeLM-33B-v1.0',
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                              use_fast=False,),
        max_out_len=1024,
        max_seq_len=4096,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=4, num_procs=1),
    )]
