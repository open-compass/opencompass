from opencompass.models import HuggingFaceCausalLM


_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='USER: ', end=' '),
        dict(role='BOT', begin='ASSISTANT: ', end='</s>', generate=True),
    ],
)

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='wizardlm-70b-v1.0-hf',
        path='WizardLM/WizardLM-70B-V1.0',
        tokenizer_path='WizardLM/WizardLM-70B-V1.0',
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
        run_cfg=dict(num_gpus=4, num_procs=1),
        end_str='</s>',
    )
]
