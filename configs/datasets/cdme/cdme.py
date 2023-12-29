from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.cdme.cdme import CDMEDataset,CDMEEvaluator,cdme_postprocess,cdme_dataset_postprocess
import os

cdme_reader_cfg = dict(input_columns=['prompt'], output_column='answer')

cdme_infer_cfg = dict(
    prompt_template=dict(
type=PromptTemplate,
        template=
        '''{prompt}'''),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512))

cdme_eval_cfg = dict(
    evaluator=dict(type=CDMEEvaluator),  
    pred_postprocessor=dict(type=cdme_postprocess),  
    dataset_postprocessor=dict(type=cdme_dataset_postprocess))  



base_path = './data/CDME/processed'
cdme_datasets = []

for folder in os.listdir(base_path):
    if os.path.isdir(os.path.join(base_path, folder)):
        dataset_dict = dict(
            abbr=f'CDME_{folder}',
            type=CDMEDataset,
            path=os.path.join(base_path, folder),
            reader_cfg=cdme_reader_cfg,
            infer_cfg=cdme_infer_cfg,
            eval_cfg=cdme_eval_cfg
        )
        cdme_datasets.append(dataset_dict)

