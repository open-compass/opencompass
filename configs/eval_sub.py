from mmengine.config import read_base
from opencompass.models import HuggingFaceCausalLM
from opencompass.summarizers import SubjectSummarizer

with read_base():
    # choose a list of datasets
    from .datasets.subject.corev2_judge import judge_corev2_datasets

datasets = [*judge_corev2_datasets]


meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<|User|>:', end='\n'),
        dict(role='BOT', begin='<|Bot|>:选择：', end='<eoa>\n', generate=True), ######选择：
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
    )
]

summarizer = dict(
    type=SubjectSummarizer
)

work_dir = './trash/subject/judge'