from opencompass.datasets import MedbulletsDataset, MedbulletsEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

import os

SYSTEM_PROMPT = 'You are a helpful medical assistant.\n\n' # Where to put this?
ZERO_SHOT_PROMPT = 'Q: {question}\n Please select the correct answer from the options above and output only the corresponding letter (A, B, C, D, or E) without any explanation or additional text.\n'

# Reader configuration
reader_cfg = dict(
    input_columns=[
        'question',
        'options',
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
    evaluator=dict(type=MedbulletsEvaluator),
    pred_role='BOT',
)
medbullets_dataset = dict(
    type=MedbulletsDataset,
    abbr='medbullets',
    path='opencompass/medbullets',
    prompt_mode='zero-shot',
    reader_cfg=reader_cfg,
    infer_cfg=infer_cfg,
    eval_cfg=eval_cfg,
    
)

medbullets_datasets = [medbullets_dataset]
