
import os
import json
from opencompass.datasets.korbench.counterfactual import korbenchcounterfactualDataset
from opencompass.datasets.korbench.counterfactual import korbenchcounterfactualEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

# Prompt template
prompt_template = dict(
    type=PromptTemplate,
    template=dict(
        begin=[dict(role="HUMAN", prompt="You are an expert in counterfactual. Solve the following problem.")],
        round=[dict(role="HUMAN", prompt="### Task:{prompt}")])
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
    evaluator=dict(type=korbenchcounterfactualEvaluator),
    pred_role="BOT",
)

korbench_counterfactual_dataset = dict(
    type=korbenchcounterfactualDataset,
    abbr="korbench_counterfactual",
    path="opencompass/korbench",
    reader_cfg=reader_cfg,
    infer_cfg=infer_cfg,
    eval_cfg=eval_cfg,
)