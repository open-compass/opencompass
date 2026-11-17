from opencompass.datasets import BFCLEvaluator, BFCLDataset
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

# BFCL ships the questions and the acceptable calls as two JSONL files. The
# dataset renders the function schemas and the user request into a single
# prompt, and the evaluator compares the reply with the acceptable calls.
bfcl_simple_datasets = [
    dict(
        abbr='bfcl_v3_simple',
        type=BFCLDataset,
        path='data/bfcl/BFCL_v3_simple.json',
        answer_path='data/bfcl/BFCL_v3_simple_answer.json',
        reader_cfg=dict(input_columns='prompt', output_column='gts'),
        infer_cfg=dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    round=[
                        dict(role='HUMAN', prompt='{prompt}'),
                    ], ),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer),
        ),
        eval_cfg=dict(
            evaluator=dict(type=BFCLEvaluator),
            pred_role='BOT',
        ),
    ),
]

datasets = bfcl_simple_datasets
