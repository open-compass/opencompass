from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='huatuogpt2-7b-hf',
        path='FreedomIntelligence/HuatuoGPT2-7B',
        max_out_len=1024,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1),
    )
]
