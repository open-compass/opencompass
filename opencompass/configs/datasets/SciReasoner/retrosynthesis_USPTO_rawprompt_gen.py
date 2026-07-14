from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import RetrosynthesisEvaluator, Retrosynthesis_postprocess, LLM4ChemDataset

reader_cfg = dict(input_columns=['input'], output_column='output')

infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            {'role': 'user', 'content': '{input}'},
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

eval_cfg = dict(
    evaluator=dict(type=RetrosynthesisEvaluator, beam_size=1, n_best=1),
    pred_postprocessor=dict(type=Retrosynthesis_postprocess),
    dataset_postprocessor=dict(type=Retrosynthesis_postprocess),
)

task = 'retrosynthesis_uspto50k'

Retrosynthesis_datasets = [
    dict(
        abbr='SciReasoner-retrosynthesis_USPTO_50K',
        type=LLM4ChemDataset,
        path='opencompass/SciReasoner-smol',
        task=task,
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg= eval_cfg,
    )
]
mini_Retrosynthesis_datasets = [
    dict(
        abbr='SciReasoner-retrosynthesis_USPTO_50K-mini',
        type=LLM4ChemDataset,
        path='opencompass/SciReasoner-smol',
        task=task,
        mini_set=True,
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg= eval_cfg,
    )
]