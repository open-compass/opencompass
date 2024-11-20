
import os
import json
from opencompass.datasets.korbench.operation import korbenchoperationDataset
from opencompass.datasets.korbench.operation import korbenchoperationEvaluator
from opencompass.openicl.icl_inferencer import korbench_GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

# Configuration
dataset_path = f"{os.getenv('BASE_PATH')}/data/korbench/operation/zero-shot.jsonl"
output_path = f"{os.getenv('BASE_PATH')}/outputs/matrix_scripts/operation/zero-shot"
metadata_file = f"{os.getenv('BASE_PATH')}/outputs/metadata/metadata_operation_zero-shot.json"
mode = "zero-shot"

# Ensure output directories exist
os.makedirs(output_path, exist_ok=True)

# Prompt template
prompt_template = dict(
    type=PromptTemplate,
    template=dict(
        begin=[dict(role="HUMAN", prompt="You are an expert in operation. Solve the following operation problem.")],
        round=[dict(role="HUMAN", prompt="### Operation Task::{prompt}### Answer:")])
)

# Reader configuration
reader_cfg = dict(
    input_columns=["prompt"],
    output_column="answer",
)

# Inference configuration
infer_cfg = dict(
    prompt_template=prompt_template,
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=korbench_GenInferencer, max_out_len=1024),
)

# Evaluation configuration
eval_cfg = dict(
    evaluator=dict(type=korbenchoperationEvaluator),
    pred_role="BOT",
)

korbench_operation_zero_shot_dataset = dict(
    type=korbenchoperationDataset,
    abbr="korbench_operation_zero_shot",
    path=dataset_path,
    reader_cfg=reader_cfg,
    infer_cfg=infer_cfg,
    eval_cfg=eval_cfg,
)
