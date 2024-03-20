from mmengine.config import read_base

with read_base():
    from ..datasets.subjective.multiround.mtbench_single_judge_diff_temp import subjective_datasets
    # from .datasets.subjective.multiround.mtbench_pair_judge import subjective_datasets
    from .model_cfg import models, judge_model, given_pred, infer, gpt4, runner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.partitioners.sub_size import SubjectiveSizePartitioner
from opencompass.summarizers import MTBenchSummarizer

datasets = [*subjective_datasets]

for model in models:
    if 'generation_kwargs' in model:
        if 'do_sample' in model['generation_kwargs']:
            del model['generation_kwargs']['do_sample']
            
eval = dict(
    partitioner=dict(type=SubjectiveSizePartitioner, strategy='split', max_task_size=10000, mode='singlescore', models=models),
    runner=runner
)

summarizer = dict(type=MTBenchSummarizer, judge_type='single')

work_dir = 'outputs/mtbench/'
