from opencompass.models import HuggingFaceCausalLM


models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='baichuan-13b-chat-hf',
        path='baichuan-inc/Baichuan-13B-Chat',
        tokenizer_path='baichuan-inc/Baichuan-13B-Chat',
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                              use_fast=False,),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True, revision='75cc8a7e5220715ebccb771581e6ca8c1377cf71'),
        run_cfg=dict(num_gpus=2, num_procs=1),
    )
]
