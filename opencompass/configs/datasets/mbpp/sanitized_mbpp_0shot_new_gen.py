from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import SanitizedMBPPDataset, MBPPEvaluator

sanitized_mbpp_reader_cfg = dict(input_columns=['text', 'test_list'], output_column='test_list_2')

prompt = '''
You are an expert Python programmer, and here is your task: {text} Your code should pass these tests:

{test_list}

[BEGIN]
'''.strip()

sanitized_mbpp_infer_cfg = dict(
    prompt_template=dict(type=PromptTemplate, template=prompt),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

sanitized_mbpp_eval_cfg = dict(evaluator=dict(type=MBPPEvaluator), pred_role='BOT')

sanitized_mbpp_datasets = [
    dict(
        type=SanitizedMBPPDataset,
        abbr='sanitized_mbpp',
        path='opencompass/sanitized_mbpp',
        reader_cfg=sanitized_mbpp_reader_cfg,
        infer_cfg=sanitized_mbpp_infer_cfg,
        eval_cfg=sanitized_mbpp_eval_cfg,
    )
]
