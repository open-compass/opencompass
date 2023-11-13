from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets.reasonbench import AnlgDataset, AnliDataset, corr2causeDataset, BBH_LogicalDeduction_Dataset, BBH_ObjectCounting_Dataset, OCNLIDataset, SIQADataset, DEERDataset

reasonbench_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role="BOT",
)

reader_cfgs, infer_cfgs = [], []
for i in range(2, 5):
    choices = ["A", "B", "C", "D"][:i]

    reader_cfgs.append(dict(
    input_columns=["prompt_ppl"] + choices + ["choices"],
    output_column="label")
    )

    infer_cfgs.append(dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            str(id):
            dict(
                round=[
                    dict(role="HUMAN", prompt="{prompt_ppl}Answer:"),
                    dict(role="BOT", prompt=f"{choice}")
                ], )
            for id, choice in enumerate(choices)
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer)
    ))

anlg = [
    dict(
        abbr="reasonbench-Anlg",
        type=AnlgDataset,
        path="data/reasonbench/Anlg.jsonl",
        reader_cfg=reader_cfgs[0],
        infer_cfg=infer_cfgs[0],
        eval_cfg=reasonbench_eval_cfg),
]

anli = [
    dict(
        abbr="reasonbench-Anli",
        type=AnliDataset,
        path="data/reasonbench/Anli.jsonl",
        reader_cfg=reader_cfgs[0],
        infer_cfg=infer_cfgs[0],
        eval_cfg=reasonbench_eval_cfg),
]

corr2cause = [
    dict(
        abbr="reasonbench-corr2cause",
        type=corr2causeDataset,
        path="data/reasonbench/corr2cause.jsonl",
        reader_cfg=reader_cfgs[0],
        infer_cfg=infer_cfgs[0],
        eval_cfg=reasonbench_eval_cfg),
]

BBH_ObjectCounting = [
    dict(
        abbr="reasonbench-BBH_ObjectCounting_Dataset",
        type=BBH_ObjectCounting_Dataset,
        path="data/reasonbench/BBH-ObjectCounting.jsonl",
        reader_cfg=reader_cfgs[2],
        infer_cfg=infer_cfgs[2],
        eval_cfg=reasonbench_eval_cfg),
]

BBH_LogicalDeduction = [
    dict(
        abbr="reasonbench-BBH_LogicalDeduction_Dataset",
        type=BBH_LogicalDeduction_Dataset,
        path="data/reasonbench/BBH-LogicalDeduction.jsonl",
        reader_cfg=reader_cfgs[1],
        infer_cfg=infer_cfgs[1],
        eval_cfg=reasonbench_eval_cfg),
]

OCNLI = [
    dict(
        abbr="reasonbench-OCNLIDataset",
        type=OCNLIDataset,
        path="data/reasonbench/OCNLI.jsonl",
        reader_cfg=reader_cfgs[1],
        infer_cfg=infer_cfgs[1],
        eval_cfg=reasonbench_eval_cfg),
]

SIQA = [
    dict(
        abbr="reasonbench-SIQA",
        type=SIQADataset,
        path="data/reasonbench/SIQA.jsonl",
        reader_cfg=reader_cfgs[1],
        infer_cfg=infer_cfgs[1],
        eval_cfg=reasonbench_eval_cfg),
]

DEER = [
    dict(
        abbr="reasonbench-DEER",
        type=DEERDataset,
        path="data/reasonbench/DEER.jsonl",
        reader_cfg=reader_cfgs[0],
        infer_cfg=infer_cfgs[0],
        eval_cfg=reasonbench_eval_cfg),
]

reasonbench_datasets = anlg + anli + corr2cause + BBH_ObjectCounting + BBH_LogicalDeduction + OCNLI + SIQA + DEER
