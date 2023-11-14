from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.utils.text_postprocessors import first_capital_postprocess
from opencompass.datasets.reasonbench import TwoOptionDataset, ThreeOptionDataset, FourOptionDataset

reasonbench_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_postprocessor=dict(type=first_capital_postprocess)
)

reader_cfgs = []
for i in range(2, 5):
    choices = ["A", "B", "C", "D"][:i]

    reader_cfgs.append(dict(
    input_columns=["prompt_ppl"],
    output_column="label_ppl")
    )

infer_cfg=dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(
            begin="</E>",
            round=[
                dict(
                    role="HUMAN",
                    prompt="</E>{prompt_ppl}"
                ),
                dict(role="BOT", prompt="Answer: {label_ppl}"),
            ]),
        ice_token="</E>",
        ),
    retriever=dict(type=FixKRetriever, fix_id_list=[]),
    inferencer=dict(type=GenInferencer)
)


CausalReasoningDataset = [
    dict(
        abbr="reasonbench-causal",
        type=TwoOptionDataset,
        path="data/reasonbench/causal.jsonl",
        reader_cfg=reader_cfgs[0],
        infer_cfg=infer_cfg,
        eval_cfg=reasonbench_eval_cfg),
]

CommonsenseReasoningDataset = [
    dict(
        abbr="reasonbench-commonsense",
        type=ThreeOptionDataset,
        path="data/reasonbench/commonsense.jsonl",
        reader_cfg=reader_cfgs[1],
        infer_cfg=infer_cfg,
        eval_cfg=reasonbench_eval_cfg),
]

AbductiveReasoningDataset = [
    dict(
        abbr="reasonbench-abductive",
        type=TwoOptionDataset,
        path="data/reasonbench/abductive.jsonl",
        reader_cfg=reader_cfgs[0],
        infer_cfg=infer_cfg,
        eval_cfg=reasonbench_eval_cfg),
]

DeductiveReasoningDataset = [
    dict(
        abbr="reasonbench-deductive",
        type=ThreeOptionDataset,
        path="data/reasonbench/deductive.jsonl",
        reader_cfg=reader_cfgs[1],
        infer_cfg=infer_cfg,
        eval_cfg=reasonbench_eval_cfg),
]

InductiveReasoningDataset = [
    dict(
        abbr="reasonbench-inductive",
        type=TwoOptionDataset,
        path="data/reasonbench/inductive.jsonl",
        reader_cfg=reader_cfgs[0],
        infer_cfg=infer_cfg,
        eval_cfg=reasonbench_eval_cfg),
]

SymbolicReasoningDataset = [
    dict(
        abbr="reasonbench-symbolic",
        type=FourOptionDataset,
        path="data/reasonbench/symbolic.jsonl",
        reader_cfg=reader_cfgs[2],
        infer_cfg=infer_cfg,
        eval_cfg=reasonbench_eval_cfg),
]

own_CommonsenseReasoningDataset = [
    dict(
        abbr="reasonbench-own_commonsense",
        type=ThreeOptionDataset,
        path="data/reasonbench/cleva_commonsense.jsonl",
        reader_cfg=reader_cfgs[1],
        infer_cfg=infer_cfg,
        eval_cfg=reasonbench_eval_cfg),
]

own_DeductiveReasoningDataset = [
    dict(
        abbr="reasonbench-own_deductive",
        type=ThreeOptionDataset,
        path="data/reasonbench/cleva_deductive.jsonl",
        reader_cfg=reader_cfgs[1],
        infer_cfg=infer_cfg,
        eval_cfg=reasonbench_eval_cfg),
]

own_InductiveReasoningDataset = [
    dict(
        abbr="reasonbench-own_inductive",
        type=TwoOptionDataset,
        path="data/reasonbench/cleva_inductive.jsonl",
        reader_cfg=reader_cfgs[0],
        infer_cfg=infer_cfg,
        eval_cfg=reasonbench_eval_cfg),
]

reasonbench_datasets = own_CommonsenseReasoningDataset + own_DeductiveReasoningDataset + own_InductiveReasoningDataset + CausalReasoningDataset + CommonsenseReasoningDataset + AbductiveReasoningDataset + DeductiveReasoningDataset + InductiveReasoningDataset + SymbolicReasoningDataset
