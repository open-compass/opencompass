from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets.reasonbench import TwoOptionDataset, ThreeOptionDataset, FourOptionDataset

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

CausalReasoningDataset = [
    dict(
        abbr="reasonbench-causal",
        type=TwoOptionDataset,
        path="data/reasonbench/causal.jsonl",
        reader_cfg=reader_cfgs[0],
        infer_cfg=infer_cfgs[0],
        eval_cfg=reasonbench_eval_cfg),
]

CommonsenseReasoningDataset = [
    dict(
        abbr="reasonbench-commonsense",
        type=ThreeOptionDataset,
        path="data/reasonbench/commonsense.jsonl",
        reader_cfg=reader_cfgs[1],
        infer_cfg=infer_cfgs[1],
        eval_cfg=reasonbench_eval_cfg),
]

AbductiveReasoningDataset = [
    dict(
        abbr="reasonbench-abductive",
        type=TwoOptionDataset,
        path="data/reasonbench/abductive.jsonl",
        reader_cfg=reader_cfgs[0],
        infer_cfg=infer_cfgs[0],
        eval_cfg=reasonbench_eval_cfg),
]

DeductiveReasoningDataset = [
    dict(
        abbr="reasonbench-deductive",
        type=ThreeOptionDataset,
        path="data/reasonbench/deductive.jsonl",
        reader_cfg=reader_cfgs[1],
        infer_cfg=infer_cfgs[1],
        eval_cfg=reasonbench_eval_cfg),
]

InductiveReasoningDataset = [
    dict(
        abbr="reasonbench-inductive",
        type=TwoOptionDataset,
        path="data/reasonbench/inductive.jsonl",
        reader_cfg=reader_cfgs[0],
        infer_cfg=infer_cfgs[0],
        eval_cfg=reasonbench_eval_cfg),
]

SymbolicReasoningDataset = [
    dict(
        abbr="reasonbench-symbolic",
        type=FourOptionDataset,
        path="data/reasonbench/symbolic.jsonl",
        reader_cfg=reader_cfgs[2],
        infer_cfg=infer_cfgs[2],
        eval_cfg=reasonbench_eval_cfg),
]

own_CommonsenseReasoningDataset = [
    dict(
        abbr="reasonbench-own_commonsense",
        type=ThreeOptionDataset,
        path="data/reasonbench/cleva_commonsense.jsonl",
        reader_cfg=reader_cfgs[1],
        infer_cfg=infer_cfgs[1],
        eval_cfg=reasonbench_eval_cfg),
]

own_DeductiveReasoningDataset = [
    dict(
        abbr="reasonbench-own_deductive",
        type=ThreeOptionDataset,
        path="data/reasonbench/cleva_deductive.jsonl",
        reader_cfg=reader_cfgs[1],
        infer_cfg=infer_cfgs[1],
        eval_cfg=reasonbench_eval_cfg),
]

own_InductiveReasoningDataset = [
    dict(
        abbr="reasonbench-own_inductive",
        type=TwoOptionDataset,
        path="data/reasonbench/cleva_inductive.jsonl",
        reader_cfg=reader_cfgs[0],
        infer_cfg=infer_cfgs[0],
        eval_cfg=reasonbench_eval_cfg),
]

reasonbench_datasets = own_CommonsenseReasoningDataset + own_DeductiveReasoningDataset + own_InductiveReasoningDataset + CausalReasoningDataset + CommonsenseReasoningDataset + AbductiveReasoningDataset + DeductiveReasoningDataset + InductiveReasoningDataset + SymbolicReasoningDataset
