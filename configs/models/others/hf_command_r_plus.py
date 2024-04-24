from opencompass.models import HuggingFaceCausalLM

_meta_template = dict(
    round=[
        dict(role="HUMAN", begin='<|START_OF_TURN_TOKEN|><|USER_TOKEN|>', end='<|END_OF_TURN_TOKEN|>'),
        dict(role="BOT", begin="<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>", end='<|END_OF_TURN_TOKEN|>', generate=True),
    ],
)

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='command-r-plus-hf',
        path="CohereForAI/c4ai-command-r-plus",
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left', trust_remote_code=True),
        meta_template=_meta_template,
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        run_cfg=dict(num_gpus=8, num_procs=1),
        end_str='<|END_OF_TURN_TOKEN|>',
        batch_padding=True,
    )
]
