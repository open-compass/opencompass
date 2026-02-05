from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask


with read_base():
    # If you want to evaluate the full scireasoner dataset (more than one million samples)
    from opencompass.configs.datasets.SciReasoner.scireasoner_gen import scireasoner_full_datasets

    # If you only want to evaluate the miniset
    from opencompass.configs.datasets.SciReasoner.scireasoner_gen import scireasoner_mini_datasets

    from opencompass.configs.summarizers.scireasoner import SciReasonerSummarizer


datasets = sum(
    (v for k, v in locals().items() if k.endswith('_datasets')),
    [],
)

summarizer = dict(
    type=SciReasonerSummarizer,
    mini_set=False,  # When evaluating miniset, please set True
    show_details=False  # Whether you want to see the detailed results for each subset
)

system_prompt = [
    dict(
        role='SYSTEM',
        prompt='You are a professional science expert, able to reason across science fields. You answer scientific questions by integrating theory, empirical evidence, and quantitative reasoning. Provide responses that are accurate, well-justified, and as concise as possible, and clearly distinguish established facts from assumptions, approximations, and remaining uncertainties.',
    ),
]

judge_cfg = () # Config your judge model here.

for item in datasets:
    item['infer_cfg']['prompt_template']['template']['round'] = system_prompt + item['infer_cfg']['prompt_template']['template']['round']
    if 'judge_cfg' in item['eval_cfg']['evaluator']:
        item['eval_cfg']['evaluator']['judge_cfg'] = judge_cfg
    elif 'judge_cfg' in item['eval_cfg']['evaluator']['llm_evaluator']:
        item['eval_cfg']['evaluator']['llm_evaluator']['judge_cfg'] = judge_cfg


infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=8),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16,
        task=dict(type=OpenICLInferTask),
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner, n=10),
    runner=dict(
        type=LocalRunner,
         max_num_workers=16,
        task=dict(type=OpenICLEvalTask)
    ),
)


work_dir = './outputs/eval_scireasoner'


