from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.utils.text_postprocessors import first_option_postprocess
from opencompass.datasets.MedQA import MedQADataset

MedQA_reader_cfg = dict(
    input_columns=['question', 'A', 'B', 'C', 'D', 'choices'],
    output_column='label',
    test_split='validation')

MedQA_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='\nQuestion: {question}\n{choices}\nAnswer:'
                )
            ], ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

MedQA_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD')
)

MedQA_datasets = [
    dict(
        abbr='MedQA',
        type=MedQADataset,
        path='opencompass/MedQA',
        reader_cfg=MedQA_reader_cfg,
        infer_cfg=MedQA_infer_cfg,
        eval_cfg=MedQA_eval_cfg)
]
