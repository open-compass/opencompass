from opencompass.datasets.supergpqa.supergpqa import SuperGPQADataset, SuperGPQAEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever


supergpqa_0shot_single_datasets = []
prompt_template = dict(
    type=PromptTemplate,
    template=dict(
        begin=[
            dict(
                role='HUMAN',
                prompt=''
            )
        ],
        round=[
            dict(
                role='HUMAN',
                prompt='{infer_prompt}' # f-string
            )
        ]
    )
)

# Reader configuration
reader_cfg = dict(
    input_columns=['infer_prompt'],
    output_column='answer_letter',
)

# Inference configuration
infer_cfg = dict(
    prompt_template=prompt_template,
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=1024),
)

# Evaluation configuration
eval_cfg = dict(
    evaluator=dict(type=SuperGPQAEvaluator),
    pred_role='BOT',
)
supergpqa_dataset = dict(
    type=SuperGPQADataset,
    abbr='supergpqa',
    path='opencompass/supergpqa',
    prompt_mode='zero-shot',
    reader_cfg=reader_cfg,
    infer_cfg=infer_cfg,
    eval_cfg=eval_cfg,
)
# print(type(supergpqa_0shot_single_datasets))

supergpqa_0shot_single_datasets.append(supergpqa_dataset)
