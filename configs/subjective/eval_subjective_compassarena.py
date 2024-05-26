from os import getenv as gv
from opencompass.models import HuggingFaceCausalLM
from mmengine.config import read_base

with read_base():
    from ..datasets.subjective.compassarena.compassarena_compare import subjective_datasets
    from .model_cfg import models, judge_model, given_pred, infer, gpt4, runner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.partitioners.sub_size import SubjectiveSizePartitioner
from opencompass.summarizers import CompassArenaSummarizer
datasets = [*subjective_datasets]

eval = dict(
    partitioner=dict(
        type=SubjectiveSizePartitioner,
        strategy='split',
        max_task_size=10000,
        mode='m2n',
        base_models=[gpt4],
        compare_models=models,
    ),
runner=runner,
given_pred=given_pred
)

work_dir = 'outputs/compass_arena/'

summarizer = dict(type=CompassArenaSummarizer, summary_type='half_add')
