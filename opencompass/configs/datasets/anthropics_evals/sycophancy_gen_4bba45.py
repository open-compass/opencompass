from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import SycophancyDataset
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.utils.text_postprocessors import first_option_postprocess

sycophancy_reader_cfg = dict(
    input_columns=['question'],
    output_column='answer_matching_behavior',
    train_split='train',
    test_split='train')

sycophancy_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(role='HUMAN', prompt="""{question}"""),
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

sycophancy_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCDEFG'),
)

# Datasets can be downloaded from
# https://github.com/anthropics/evals/tree/main/sycophancy  # noqa
# You can choose whatever subset you need.
_dataset_list = [
    'sycophancy_on_nlp_survey',
    'sycophancy_on_philpapers2020',
    'sycophancy_on_political_typology_quiz',
]

sycophancy_datasets = []
for _dataset in _dataset_list:
    sycophancy_datasets.append(
        dict(
            abbr=f'sycophancy_{_dataset}',
            type=SycophancyDataset,
            path=f'data/sycophancy/{_dataset}.jsonl',
            reader_cfg=sycophancy_reader_cfg,
            infer_cfg=sycophancy_infer_cfg,
            eval_cfg=sycophancy_eval_cfg,
        ))
