from mmengine.config import read_base
with read_base():
    from .datasets.subjectivity_cmp.subjectivity_cmp import subjectivity_datasets
    from .summarizers.subjective import summarizer

datasets = [*subjectivity_datasets]

from opencompass.models import HuggingFaceCausalLM, HuggingFace, OpenAI
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks.subjective_eval import SubjectiveEvalTask

_meta_template = dict(
    round=[
        dict(role="HUMAN", begin='\n<|im_start|>user\n', end='<|im_end|>'),
        dict(
            role="BOT",
            begin="\n<|im_start|>assistant\n",
            end='<|im_end|>',
            generate=True),
    ], )

_meta_template2 = dict(
    round=[
        dict(role='HUMAN', begin='<|User|>:', end='<eoh>\n'),
        dict(role='BOT', begin='<|Bot|>:', end='<eoa>\n', generate=True),
    ], )

models = [
    dict(
        type=HuggingFace,
        abbr='chatglm2-6b-hf',
        path='THUDM/chatglm2-6b',
        tokenizer_path='THUDM/chatglm2-6b',
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
            revision='b1502f4f75c71499a3d566b14463edd62620ce9f'),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(
            trust_remote_code=True,
            device_map='auto',
            revision='b1502f4f75c71499a3d566b14463edd62620ce9f'),
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
    dict(
        type=HuggingFaceCausalLM,
        abbr='qwen-7b-chat-hf',
        path="/mnt/petrelfs/share_data/duanhaodong/Qwen-7B-Chat",
        tokenizer_path='/mnt/petrelfs/share_data/duanhaodong/Qwen-7B-Chat',
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
            use_fast=False,
        ),
        pad_token_id=151643,
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        meta_template=_meta_template,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
    dict(
        type=HuggingFaceCausalLM,
        abbr='internlm-chat-7b-hf',
        path="internlm/internlm-chat-7b",
        tokenizer_path='internlm/internlm-chat-7b',
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            use_fast=False,
            trust_remote_code=True,
            revision="ed5e35564ac836710817c51e8e8d0a5d4ff03102"),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        meta_template=_meta_template2,
        model_kwargs=dict(
            trust_remote_code=True,
            device_map='auto',
            revision="ed5e35564ac836710817c51e8e8d0a5d4ff03102"),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True)
    ],
    reserved_roles=[
        dict(role='SYSTEM', api_role='SYSTEM'),
    ],
)

eval = dict(
    partitioner=dict(
        type=SubjectiveNaivePartitioner,
        mode='all',  # 新参数
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=2,  # 支持并行比较
        task=dict(
            type=SubjectiveEvalTask,  # 新 task，用来读入一对 model 的输入
            judge_cfg=dict(
                abbr='GPT4',
                type=OpenAI,
                path='gpt-4-0613',
                key='ENV',
                meta_template=api_meta_template,
                query_per_second=1,
                max_out_len=2048,
                max_seq_len=2048,
                batch_size=2),
        )),
)
