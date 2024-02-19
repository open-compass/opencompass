from opencompass.models import HuggingFaceCausalLM


_meta_template = dict(
    round=[
        dict(role='HUMAN', end='\n\n'),
        dict(role='BOT', begin='### Response:', end='</s>', generate=True),
    ],
)

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='wizardmath-7b-v1.1-hf',
        path='WizardLM/WizardMath-7B-V1.1',
        tokenizer_path='WizardLM/WizardMath-7B-V1.1',
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
        ),
        meta_template=_meta_template,
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        run_cfg=dict(num_gpus=1, num_procs=1),
        end_str='</s>',
    )
]
