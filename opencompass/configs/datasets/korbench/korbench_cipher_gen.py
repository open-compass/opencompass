
import os
import json
from opencompass.datasets.korbench.cipher import korbenchcipherDataset
from opencompass.datasets.korbench.cipher import korbenchcipherEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever


# Prompt template
prompt_template = dict(
    type=PromptTemplate,
    template=dict(
        begin=[dict(role="HUMAN", prompt="You are an expert in cipher. Solve the following cipher problem.")],
        round=[dict(role="HUMAN", prompt="### Cipher Task::{prompt}")])
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
    inferencer=dict(type=GenInferencer, max_out_len=1024),
)

# Evaluation configuration
eval_cfg = dict(
    evaluator=dict(type=korbenchcipherEvaluator),
    pred_role="BOT",
)

korbench_cipher_dataset = dict(
    type=korbenchcipherDataset,
    abbr="korbench_cipher",
    path="opencompass/korbench",
    reader_cfg=reader_cfg,
    infer_cfg=infer_cfg,
    eval_cfg=eval_cfg,
)