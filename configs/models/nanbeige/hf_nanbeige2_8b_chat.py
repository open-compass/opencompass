from opencompass.models import HuggingFaceCausalLM

_meta_template = dict(
    begin="<|im_start|>system\n你是一个名为\"南北阁\"的人工智能助手，正在与人类用户进行交谈。你的目标是以最有帮助和最逻辑的方式回答问题，同时确保内容的安全性。你的回答中不应包含任何有害、政治化、宗教化、不道德、种族主义、非法的内容。请确保你的回答不带有社会偏见，符合社会主义价值观。如果遇到的问题无意义或事实上不连贯，请不要回答错误的内容，而是解释问题为何无效或不连贯。如果你不知道问题的答案，也请勿提供错误的信息。<|im_end|>\n",
    round=[
        dict(role='HUMAN', begin='<|im_start|>user\n', end='<|im_end|>\n'),
        dict(role='BOT', begin='<|im_start|>assistant\n', end='<|im_end|>\n', generate=True),
    ],
)

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='nanbeige2-8b-chat-hf',
        path="Nanbeige/Nanbeige2-8B-Chat",
        tokenizer_path='Nanbeige/Nanbeige2-8B-Chat',
        model_kwargs=dict(
            device_map='auto',
            torch_dtype='auto',
            trust_remote_code=True,
        ),
        tokenizer_kwargs=dict(
            padding_side='right',
            truncation_side='left',
            trust_remote_code=True,
            use_fast=False,
        ),
        meta_template=_meta_template,
        batch_padding=False,
        max_out_len=100,
        max_seq_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=1, num_procs=1),
        end_str='<|im_end|>',
    )
]
