from opencompass.datasets.korbench.korbench_mixed import korbenchmixedDataset, korbenchmixedEvaluator

from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
korbench_mixed_datasets = []

modes = ["Multi-Q", "Multi-R", "Multi-RQ"]  # Define available modes for mixed mode

for mode in modes:
    # Prompt template
    prompt_template = dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(
                    role="HUMAN",
                    prompt=f"You are an expert in solving tasks that require integrating multiple rules and questions. The following task is in {mode} mode."
                )
            ],
            round=[
                dict(
                    role="HUMAN",
                    prompt=f"### Mixed Task (Mode: {mode})::{{prompt}}" # f-string
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
        evaluator=dict(type=korbenchmixedEvaluator),
        pred_role="BOT",
    )

    korbench_dataset = dict(
        type=korbenchmixedDataset,
        abbr=f"korbench_mixed_{mode}",
        path="opencompass/korbench",
        mode=mode,
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
    )

    korbench_mixed_datasets.append(korbench_dataset)
