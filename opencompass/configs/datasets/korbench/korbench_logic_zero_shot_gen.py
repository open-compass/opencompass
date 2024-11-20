
import os
import json
from opencompass.datasets.korbench.logic import korbenchlogicDataset
from opencompass.datasets.korbench.logic import korbenchlogicEvaluator
from opencompass.openicl.icl_inferencer import korbench_GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

# Configuration
dataset_path = f"{os.getenv('BASE_PATH')}/data/korbench/logic/zero-shot.jsonl"
output_path = f"{os.getenv('BASE_PATH')}/outputs/matrix_scripts/logic/zero-shot"
metadata_file = f"{os.getenv('BASE_PATH')}/outputs/metadata/metadata_logic_zero-shot.json"
mode = "zero-shot"

# Ensure output directories exist
os.makedirs(output_path, exist_ok=True)

# Prompt template
prompt_template = dict(
    type=PromptTemplate,
    template=dict(
        begin=[dict(role="HUMAN", prompt="You are an expert in logic. Solve the following logic problem.")],
        round=[dict(role="HUMAN", prompt="### Logic Task::{prompt}### Answer:")])
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
    evaluator=dict(type=korbenchlogicEvaluator),
    pred_role="BOT",
)

korbench_logic_zero_shot_dataset = dict(
    type=korbenchlogicDataset,
    abbr="korbench_logic_zero_shot",
    path=dataset_path,
    reader_cfg=reader_cfg,
    infer_cfg=infer_cfg,
    eval_cfg=eval_cfg,
)
