from mmengine.config import read_base

with read_base():
    from ..datasets.subjective.alignbench.alignbench_judgeby_critiquellm import subjective_datasets
    from .model_cfg import models, judge_model, given_pred, infer, gpt4, runner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.partitioners.sub_size import SubjectiveSizePartitioner
from opencompass.summarizers import AlignmentBenchSummarizer

# -------------Inference Stage ----------------------------------------
# For subjective evaluation, we often set do sample for models
datasets = [*subjective_datasets]
eval = dict(
    partitioner=dict(
        type=SubjectiveNaivePartitioner, mode='singlescore', models=models
    ),
    runner=runner,
)

summarizer = dict(type=AlignmentBenchSummarizer, judge_type='general')
work_dir = 'outputs/alignment_bench/'
