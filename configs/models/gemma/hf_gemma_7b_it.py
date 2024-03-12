from opencompass.models import HuggingFaceCausalLM

_meta_template = dict(
    round=[
        dict(role="HUMAN", begin='<start_of_turn>user\n', end='<end_of_turn>\n'),
        dict(role="BOT", begin="<start_of_turn>model\n", end='<end_of_turn>\n', generate=True),
    ],
    eos_token_id=151645,
)

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='gemma-7b-it-hf',
        path="google/gemma-7b-it",
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
            use_fast=False,
        ),
        meta_template=_meta_template,
        min_out_len=1,
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
