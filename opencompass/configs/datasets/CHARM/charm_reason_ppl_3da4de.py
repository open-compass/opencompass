import os

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.datasets import CharmDataset
from opencompass.openicl.icl_evaluator import AccwithDetailsEvaluator

charm_tasks = [
    ['Chinese_Anachronisms_Judgment', 'AB'],
    ['Chinese_Movie_and_Music_Recommendation', 'ABCD'],
    ['Chinese_Natural_Language_Inference', 'ABC'],
    ['Chinese_Reading_Comprehension', 'ABCD'],
    ['Chinese_Sequence_Understanding', 'ABCD'],
    ['Chinese_Sport_Understanding', 'AB'],
    ['Chinese_Time_Understanding', 'ABCD'],
    ['Global_Anachronisms_Judgment', 'AB'],
    ['Global_Movie_and_Music_Recommendation', 'ABCD'],
    ['Global_Natural_Language_Inference', 'ABC'],
    ['Global_Reading_Comprehension', 'ABCD'],
    ['Global_Sequence_Understanding', 'ABCD'],
    ['Global_Sport_Understanding', 'AB'],
    ['Global_Time_Understanding', 'ABCDEF'],
]

charm_reason_datasets = []
for task_name, options in charm_tasks:

    with open(os.path.join(os.path.dirname(__file__), 'few-shot-examples', f'{task_name}_Direct.txt'), 'r') as f:
        few_shot_example = f.read()

    charm_reason_reader_cfg = dict(input_columns=['input'], output_column='target')

    charm_reason_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template={
                f'({opt})': f'{few_shot_example}\n{{input}}\nA: {opt}' for opt in options
            },
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=PPLInferencer),
    )

    charm_reason_eval_cfg = dict(evaluator=dict(type=AccwithDetailsEvaluator))

    charm_reason_datasets.append(
        dict(
            type=CharmDataset,
            abbr=f'charm-reason-{task_name}_Direct',
            path=f'data/CHARM/reasoning',
            name=task_name,
            reader_cfg=charm_reason_reader_cfg,
            infer_cfg=charm_reason_infer_cfg,
            eval_cfg=charm_reason_eval_cfg,
        )
    )
