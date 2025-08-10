from opencompass.models import HuggingFaceCausalLM

models = [
    # WizardCoder Python 34B
    dict(
        type=HuggingFaceCausalLM,
        abbr='WizardCoder-Python-34B-V1.0',
        path='WizardLM/WizardCoder-Python-34B-V1.0',
        tokenizer_path='WizardLM/WizardCoder-Python-34B-V1.0',
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
        ),
        max_out_len=1024,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(trust_remote_code=True, device_map='auto'),
        run_cfg=dict(num_gpus=4, num_procs=1),
    ),
]
