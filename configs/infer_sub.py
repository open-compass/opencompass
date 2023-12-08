from mmengine.config import read_base
from opencompass.models import HuggingFaceCausalLM

with read_base():
    # choose a list of datasets
    #from .models.qwen.hf_qwen_7b_chat import models
    from .datasets.subject.corev2_infer import infer_corev2_datasets
    # and output the results in a choosen format
    from .summarizers.medium import summarizer

datasets = [*infer_corev2_datasets]


meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<|User|>:', end='\n'),
        dict(role='BOT', begin='<|Bot|>:', end='<eoa>\n', generate=True),
    ],
    eos_token_id=103028)

models = [
dict(
        type=HuggingFaceCausalLM,
        abbr='internlm-chat-7b-hf-v11',
        path="internlm/internlm-chat-7b-v1_1",
        tokenizer_path='internlm/internlm-chat-7b-v1_1',
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            use_fast=False,
            trust_remote_code=True,
        ),
        max_out_len=512,
        max_seq_len=2048,
        batch_size=8,
        meta_template=meta_template,
        model_kwargs=dict(trust_remote_code=True, device_map='auto'),
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
dict(
        type=HuggingFaceCausalLM,
        abbr='internlm-chat-20b-hf',
        path="internlm/internlm-chat-20b",
        tokenizer_path='internlm/internlm-chat-20b',
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            use_fast=False,
            trust_remote_code=True,
        ),
        max_out_len=512,
        max_seq_len=2048,
        batch_size=8,
        meta_template=meta_template,
        model_kwargs=dict(trust_remote_code=True, device_map='auto'),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]

work_dir = './trash/subject/infer'