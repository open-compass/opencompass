from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import TheoremQADataset, TheoremQA_postprocess

TheoremQA_reader_cfg = dict(input_columns=['Question', 'Answer_type'], output_column='Answer', train_split='test')

TheoremQA_prompt1 = (
    'Please read a math problem, and then think step by step to derive the answer. The answer is decided by Answer Type. '
    'If the Answer type in [bool], the answer needs to be True or False. '
    'Else if the Answer type in [integer, float] , The answer needs to be in numerical form. '
    'Else if the Answer type in [list of integer, list of float] , the answer needs to be a list of number like [2, 3, 4]. '
    'Else if the Answer type in [option], the answer needs to be an option like (a), (b), (c), (d).'
    "You need to output the answer in your final sentence like 'Therefore, the answer is ...'."
)
TheoremQA_prompt2 = (
    f'Below is an instruction that describes a task, paired with an input that provides further context. '
    f'Write a response that appropriately completes the request.\n\n### Instruction:\n{TheoremQA_prompt1}\n\n### Input:\n{{Question}}\nAnswer_type:{{Answer_type}}\n### Response:\n'
)

TheoremQA_infer_cfg = dict(
    prompt_template=dict(type=PromptTemplate, template=TheoremQA_prompt2),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512),
)

TheoremQA_eval_cfg = dict(evaluator=dict(type=AccEvaluator), pred_postprocessor=dict(type=TheoremQA_postprocess))

TheoremQA_datasets = [
    dict(
        abbr='TheoremQA',
        type=TheoremQADataset,
        path='./data/TheoremQA/test.csv',
        reader_cfg=TheoremQA_reader_cfg,
        infer_cfg=TheoremQA_infer_cfg,
        eval_cfg=TheoremQA_eval_cfg,
    )
]
