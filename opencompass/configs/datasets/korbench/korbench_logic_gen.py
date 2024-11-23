import os
import json
from opencompass.datasets.korbench.korbench_single_0shot import korbenchsingle0shotDataset, korbenchsingle0shotEvaluator

from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

categories = ["cipher", "counterfactual", "logic", "operation", "puzzle"]

korbench_0shot_single_datasets = []

for category in categories:
    # Prompt template
    prompt_template = dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(
                    role="HUMAN",
                    prompt=f"You are an expert in {category}. Solve the following {category} problem."
                )
            ],
            round=[
                dict(
                    role="HUMAN",
                    prompt=f"### {category} Task::{{prompt}}" # f-string
                )
            ]
        )
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
        evaluator=dict(type=korbenchsingle0shotEvaluator),
        pred_role="BOT",
    )

    korbench_dataset = dict(
        type=korbenchsingle0shotDataset,
        abbr=f"korbench_{category}",
        path="opencompass/korbench",
        category=category,
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
    )

    korbench_0shot_single_datasets.append(korbench_dataset)
