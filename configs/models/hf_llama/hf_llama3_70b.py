from opencompass.models import HuggingFaceCausalLM


models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr="llama-3-70b-hf",
        path="meta-llama/Meta-Llama-3-70B",
        model_kwargs=dict(device_map="auto"),
        tokenizer_kwargs=dict(
            padding_side="left",
            truncation_side="left",
            use_fast=False,
        ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        batch_padding=True,
        run_cfg=dict(num_gpus=4, num_procs=1),
    )
]
