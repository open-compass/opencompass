from opencompass.datasets import SciKnowEvalDataset, SciKnowEvalEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

ZERO_SHOT_PROMPT = '{q4}'

# Reader configuration
reader_cfg = dict(
    input_columns=[
        'prompt',
        'question',
        'choices',
        'label',
        'answerKey',
        'type',
        'domain',
        'details',
        'answer',
        'q4'
    ],
    output_column='answerKey',
)

# Inference configuration
infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt=ZERO_SHOT_PROMPT, # prompt mode: zero-shot
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

# Evaluation configuration
eval_cfg = dict(
    evaluator=dict(type=SciKnowEvalEvaluator),
    pred_role='BOT',
)
sciknoweval_dataset_biology = dict(
    type=SciKnowEvalDataset,
    abbr='sciknoweval_biology',
    path='hicai-zju/SciKnowEval',
    prompt_mode='zero-shot',
    subset='biology',
    reader_cfg=reader_cfg,
    infer_cfg=infer_cfg,
    eval_cfg=eval_cfg,
)

sciknoweval_dataset_chemistry = dict(
    type=SciKnowEvalDataset,
    abbr='sciknoweval_chemistry',
    path='hicai-zju/SciKnowEval',
    subset='chemistry',
    prompt_mode='zero-shot',
    reader_cfg=reader_cfg,
    infer_cfg=infer_cfg,
    eval_cfg=eval_cfg,
)

sciknoweval_dataset_material = dict(
    type=SciKnowEvalDataset,
    abbr='sciknoweval_material',
    path='hicai-zju/SciKnowEval',
    subset='material',
    prompt_mode='zero-shot',
    reader_cfg=reader_cfg,
    infer_cfg=infer_cfg,
    eval_cfg=eval_cfg,
)

sciknoweval_dataset_physics = dict(
    type=SciKnowEvalDataset,
    abbr='sciknoweval_physics',
    path='hicai-zju/SciKnowEval',
    prompt_mode='zero-shot',
    subset='physics',
    reader_cfg=reader_cfg,
    infer_cfg=infer_cfg,
    eval_cfg=eval_cfg,
)


sciknoweval_datasets = [sciknoweval_dataset_biology, sciknoweval_dataset_chemistry, sciknoweval_dataset_physics, sciknoweval_dataset_material]
