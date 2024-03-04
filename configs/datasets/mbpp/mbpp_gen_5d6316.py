from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import MBPPDataset, MBPPEvaluator2

mbpp_reader_cfg = dict(input_columns=["text", "test_list"], output_column="test_list_2")

# This prompt is used for WizardLMCode series
# You can use other config file for basic 3-shot generation
mbpp_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role="HUMAN",
                    prompt="""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Create a Python script for this problem:

{text}
Test examples:
{test_list}

### Response:""",
                ),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512),
)

mbpp_eval_cfg = dict(evaluator=dict(type=MBPPEvaluator2), pred_role="BOT")

mbpp_datasets = [
    dict(
        type=MBPPDataset,
        abbr="mbpp",
        path="./data/mbpp/mbpp.jsonl",
        reader_cfg=mbpp_reader_cfg,
        infer_cfg=mbpp_infer_cfg,
        eval_cfg=mbpp_eval_cfg,
    )
]
