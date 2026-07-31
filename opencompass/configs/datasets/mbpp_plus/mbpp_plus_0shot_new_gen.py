from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import MBPPEvaluator, MBPPPlusDataset

mbpp_plus_reader_cfg = dict(
    input_columns=['text', 'test_list'], output_column='task_id')

prompt = '''
You are an expert Python programmer, and here is your task: {text} Your code should pass these tests:

{test_list}

[BEGIN]
'''.strip()

mbpp_plus_infer_cfg = dict(
    prompt_template=dict(type=PromptTemplate, template=prompt),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

mbpp_plus_eval_cfg = dict(evaluator=dict(type=MBPPEvaluator, metric='MBPPPlus'), pred_role='BOT')

mbpp_plus_datasets = [
    dict(
        type=MBPPPlusDataset,
        abbr='mbpp_plus',
        path='opencompass/mbpp_plus',
        reader_cfg=mbpp_plus_reader_cfg,
        infer_cfg=mbpp_plus_infer_cfg,
        eval_cfg=mbpp_plus_eval_cfg,
    )
]
