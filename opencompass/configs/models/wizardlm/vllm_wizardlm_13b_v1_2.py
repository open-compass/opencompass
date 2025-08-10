from opencompass.models import VLLM


_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='USER: ', end=' '),
        dict(role='BOT', begin='ASSISTANT: ', end='</s>', generate=True),
    ],
)

models = [
    dict(
        type=VLLM,
        abbr='wizardlm-13b-v1.2-vllm',
        path='WizardLM/WizardLM-13B-V1.2',
        meta_template=_meta_template,
        max_out_len=100,
        max_seq_len=2048,
        batch_size=1,
        generation_kwargs=dict(temperature=0),
        stop_words=['</s>'],
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
