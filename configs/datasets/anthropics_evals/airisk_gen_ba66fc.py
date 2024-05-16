from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import AiRiskDataset
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.utils.text_postprocessors import first_option_postprocess

airisk_reader_cfg = dict(
    input_columns=['question'],
    output_column='answer_matching_behavior',
    train_split='train',
    test_split='train')

airisk_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(role='HUMAN', prompt="""{question}"""),
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

airisk_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCDEFGH'),
)

# Datasets can be downloaded from
# https://github.com/anthropics/evals/tree/main/advanced-ai-risk/human_generated_evals  # noqa
# We only use two subset currently, `corrigible-less-HHH` and `corrigible-more-HHH`  # noqa
# because the targets of these dimensions
# are not exactly the same. You can choose whatever subset you need.
_dataset_list = [
    'coordinate-itself',
    'coordinate-other-ais',
    'coordinate-other-versions',
    'corrigible-less-HHH',
    'corrigible-more-HHH',
    'corrigible-neutral-HHH',
    'myopic-reward',
    'one-box-tendency',
    'power-seeking-inclination',
    'self-awareness-general-ai',
    'self-awareness-good-text-model',
    'self-awareness-text-model',
    'self-awareness-training-architecture',
    'self-awareness-web-gpt',
    'survival-instinct',
    'wealth-seeking-inclination',
]

airisk_datasets = []
for _dataset in _dataset_list:
    airisk_datasets.append(
        dict(
            abbr=f'airisk_{_dataset}',
            type=AiRiskDataset,
            path=
            f'data/advanced-ai-risk/human_generated_evals/{_dataset}.jsonl',
            reader_cfg=airisk_reader_cfg,
            infer_cfg=airisk_infer_cfg,
            eval_cfg=airisk_eval_cfg,
        ))
