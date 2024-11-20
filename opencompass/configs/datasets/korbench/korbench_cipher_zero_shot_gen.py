
import os
import json
from opencompass.datasets.korbench.cipher import korbenchcipherDataset
from opencompass.datasets.korbench.cipher import korbenchcipherEvaluator
from opencompass.openicl.icl_inferencer import korbench_GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

# Configuration
dataset_path = "/home/epsilon/miniforge3/my_opencompass_project/opencompass/data/korbench/cipher/zero-shot.jsonl"
output_path = "/home/epsilon/miniforge3/my_opencompass_project/opencompass/outputs/matrix_scripts/cipher/zero-shot"
metadata_file = "/home/epsilon/miniforge3/my_opencompass_project/opencompass/outputs/metadata/metadata_cipher_zero-shot.json"
mode = "zero-shot"

# Ensure output directories exist
os.makedirs(output_path, exist_ok=True)

# Prompt template
prompt_template = dict(
    type=PromptTemplate,
    template=dict(
        begin=[dict(role="HUMAN", prompt="You are an expert in cipher. Solve the following cipher problem.")],
        round=[dict(role="HUMAN", prompt="### Cipher Task::{prompt}### Answer:")])
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
    evaluator=dict(type=korbenchcipherEvaluator),
    pred_role="BOT",
)

korbench_cipher_zero_shot_dataset = dict(
    type=korbenchcipherDataset,
    abbr="korbench_cipher_zero_shot",
    path=dataset_path,
    reader_cfg=reader_cfg,
    infer_cfg=infer_cfg,
    eval_cfg=eval_cfg,
)