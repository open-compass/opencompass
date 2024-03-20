from mmengine.config import read_base

with read_base():
    from ..datasets.subjective.alpaca_eval.alpacav1_judgeby_gpt4 import subjective_datasets as alpacav1
    from ..datasets.subjective.alpaca_eval.alpacav2_judgeby_gpt4 import subjective_datasets as alpacav2
    from .model_cfg import models, judge_model, given_pred, infer, gpt4, runner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.partitioners.sub_size import SubjectiveSizePartitioner
from opencompass.summarizers import AlpacaSummarizer
datasets = [*alpacav2]
eval = dict(
    partitioner=dict(
        type=SubjectiveSizePartitioner, max_task_size=1000, mode='m2n', base_models=[gpt4], compare_models=models
    ),
runner=runner,
given_pred=given_pred
)
work_dir = 'outputs/alpaca/'

summarizer = dict(type=AlpacaSummarizer, judge_type='v2')
