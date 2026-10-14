import os

from opencompass.datasets import VLMEvalKitDataset
from opencompass.evaluator import VLMEvalKitEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

dataset_name = 'MMMU_Pro_10c'
data_root = os.getenv('LMUData', 'data/vlmevalkit')
sample_limit = int(os.getenv('MMMU_PRO_SAMPLE_LIMIT', '0')) or None
vlmeval_dataset_cfg = dict(dataset_name=dataset_name,
                           data_root=data_root,
                           dataset_kwargs=dict(),
                           sample_limit=sample_limit)

reader_cfg = dict(input_columns=['prompt'], output_column='answer')

infer_cfg = dict(prompt_template=dict(type=RawPromptTemplate,
                                      messages=[dict(expand_column='prompt')],
                                      format_variables=False),
                 retriever=dict(type=ZeroRetriever),
                 inferencer=dict(type=GenInferencer))

eval_cfg = dict(evaluator=dict(
    type=VLMEvalKitEvaluator, **vlmeval_dataset_cfg, eval_kwargs=dict()))

mmmu_pro_datasets = [
    dict(type=VLMEvalKitDataset,
         abbr=dataset_name,
         **vlmeval_dataset_cfg,
         reader_cfg=reader_cfg,
         infer_cfg=infer_cfg,
         eval_cfg=eval_cfg)
]
