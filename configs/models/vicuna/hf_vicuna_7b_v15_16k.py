from opencompass.models import HuggingFaceCausalLM

_meta_template = dict(
    round=[
        dict(role="HUMAN", begin='USER: '),
        dict(role="BOT", begin=" ASSISTANT:", end='</s>', generate=True),
    ],
)

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='vicuna-7b-v1.5-16k-hf',
        path="lmsys/vicuna-7b-v1.5-16k",
        tokenizer_path='lmsys/vicuna-7b-v1.5-16k',
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            use_fast=False,
        ),
        meta_template=_meta_template,
        max_out_len=100,
        max_seq_len=8192,
        batch_size=8,
        model_kwargs=dict(device_map='auto'),
        batch_padding=False, # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=1, num_procs=1),
        end_str='</s>',
    )
]
