from opencompass.models import HuggingFaceCausalLM

models = [
    # LLaMA 65B
    dict(
        type=HuggingFaceCausalLM,
        abbr='llama-65b-hf',
        path="huggyllama/llama-65b",
        tokenizer_path='huggyllama/llama-65b',
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False,
                              ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto'),
        batch_padding=False, # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=4, num_procs=1),
    )
]
