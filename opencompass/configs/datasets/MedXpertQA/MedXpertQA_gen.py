from opencompass.datasets import MedXpertQADataset, MedXpertQAEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

SYSTEM_PROMPT = 'You are a helpful medical assistant.\n\n' # Where to put this?
ZERO_SHOT_PROMPT = 'Q: {question}\nA: Among {start} through {end}, the answer is'

# Reader configuration
reader_cfg = dict(
    input_columns=[
        'question',
        'options',
        'medical_task',
        'body_system',
        'question_type',
        'prompt_mode',
    ],
    output_column='label',
)

# Inference configuration
infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(role='SYSTEM', fallback_role='HUMAN', prompt=SYSTEM_PROMPT),
            ],
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
    evaluator=dict(type=MedXpertQAEvaluator),
    pred_role='BOT',
)
medxpertqa_dataset = dict(
    type=MedXpertQADataset,
    abbr='medxpertqa',
    path='TsinghuaC3I/MedXpertQA',
    prompt_mode='zero-shot',
    reader_cfg=reader_cfg,
    infer_cfg=infer_cfg,
    eval_cfg=eval_cfg,
)

medxpertqa_datasets = [medxpertqa_dataset]
