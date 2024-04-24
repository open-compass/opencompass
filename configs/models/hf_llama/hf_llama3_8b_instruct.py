from opencompass.models import HuggingFaceCausalLM

_meta_template = dict(
    round=[
        dict(role="HUMAN", begin="<|start_header_id|>user<|end_header_id|>\n\n", end="<|eot_id|>"),
        dict(role="BOT", begin="<|start_header_id|>assistant<|end_header_id|>\n\n", end="<|eot_id|>", generate=True),
    ],
)

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr="llama-3-8b-instruct-hf",
        path="meta-llama/Meta-Llama-3-8B-Instruct",
        model_kwargs=dict(device_map="auto"),
        tokenizer_kwargs=dict(
            padding_side="left",
            truncation_side="left",
            use_fast=False,
        ),
        meta_template=_meta_template,
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        run_cfg=dict(num_gpus=1, num_procs=1),
        generation_kwargs={"eos_token_id": [128001, 128009]},
        batch_padding=True,
    )
]
