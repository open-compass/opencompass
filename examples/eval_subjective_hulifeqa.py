from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.OpenHuEval.HuLifeQA import (
        hu_life_qa_datasets,
        task_group_new,
    )
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_5_7b_chat import (
        models as lmdeploy_internlm2_5_7b_chat_model,
    )
    from opencompass.configs.models.openai.gpt_4o_mini_20240718 import (
        models as gpt_4o_mini_20240718_model,
    )

from opencompass.models import OpenAI
from opencompass.partitioners import (
    NumWorkerPartitioner,
    SubjectiveNumWorkerPartitioner,
)
from opencompass.runners import LocalRunner, SlurmSequentialRunner
from opencompass.summarizers import WildBenchSingleSummarizer
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask

api_meta_template = dict(
    round=[
        dict(role="SYSTEM", api_role="SYSTEM"),
        dict(role="HUMAN", api_role="HUMAN"),
        dict(role="BOT", api_role="BOT", generate=True),
    ]
)

models = [
    *gpt_4o_mini_20240718_model,
    *lmdeploy_internlm2_5_7b_chat_model,
]

judge_models = [
    dict(
        abbr="GPT-4o-2024-08-06",
        type=OpenAI,
        path="gpt-4o-2024-08-06",
        key="ENV",
        meta_template=api_meta_template,
        query_per_second=16,
        max_out_len=4096,
        max_seq_len=4096,
        batch_size=8,
        temperature=0,
    )
]

for ds in hu_life_qa_datasets:
    ds.update(
        dict(
            mode="singlescore",
            eval_mode="single"
        )
    )
del ds
datasets = [*hu_life_qa_datasets]
del hu_life_qa_datasets

infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=8,
    ),
    runner=dict(
        type=SlurmSequentialRunner,
        max_num_workers=16,
        task=dict(type=OpenICLInferTask),
    ),
)

eval = dict(
    partitioner=dict(
        type=SubjectiveNumWorkerPartitioner,
        num_worker=8,
        models=models,
        judge_models=judge_models,
    ),
    runner=dict(
        type=LocalRunner, 
        max_num_workers=16, 
        task=dict(type=SubjectiveEvalTask)
    ),
)

summarizer = dict(
    type=WildBenchSingleSummarizer,
    customized_task_group_new=task_group_new,
)

work_dir = (
    "./outputs/" + __file__.split("/")[-1].split(".")[0] + "/"
)  # do NOT modify this line, yapf: disable, pylint: disable
