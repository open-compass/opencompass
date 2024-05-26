from mmengine.config import read_base

with read_base():
    from ..datasets.subjective.alpaca_eval.alpacav1_judgeby_gpt4 import subjective_datasets as alpacav1
    from ..datasets.subjective.alpaca_eval.alpacav2_judgeby_gpt4 import subjective_datasets as alpacav2
    from .model_cfg import models, judge_model, given_pred, infer, gpt4, runner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.partitioners.sub_size import SubjectiveSizePartitioner
from opencompass.summarizers import AlpacaSummarizer
from opencompass.tasks.outer_eval.alpacaeval import AlpacaEvalTask
datasets = [*alpacav2]
gpt4_judge = dict(
    abbr='GPT4-Turbo',
    path='gpt-4-1106-preview',
    key='',  # The key will be obtained from $OPENAI_API_KEY, but you can write down your key here as well
    config='weighted_alpaca_eval_gpt4_turbo'
)
## ------------- Evaluation Configuration
eval = dict(
    partitioner=dict(
        type=NaivePartitioner
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=256,
        task=dict(type=AlpacaEvalTask, judge_cfg=gpt4_judge),
    )
)
work_dir = 'outputs/alpaca/'
