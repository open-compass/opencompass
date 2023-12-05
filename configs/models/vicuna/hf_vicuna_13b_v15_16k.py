from opencompass.models import HuggingFaceCausalLM


models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='vicuna-13b-v1.5-16k-hf',
        path="lmsys/vicuna-13b-v1.5-16k",
        tokenizer_path='lmsys/vicuna-13b-v1.5-16k',
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            use_fast=False,
        ),
        max_out_len=100,
        max_seq_len=8192,
        batch_size=8,
        model_kwargs=dict(device_map='auto'),
        batch_padding=False, # if false, inference with for-loop without batch padding
        use_fastchat_template=True,
        run_cfg=dict(num_gpus=2, num_procs=1)
    )
]
