from opencompass.models import HuggingFaceCausalLM


models = [dict(
        type=HuggingFaceCausalLM,
        abbr='pandalm-7b-v1-hf',
        path='WeOpenML/PandaLM-7B-v1',
        tokenizer_path='WeOpenML/PandaLM-7B-v1',
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
