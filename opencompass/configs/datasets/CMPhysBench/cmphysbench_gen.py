from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.cmphysbench import CMPhysBenchDataset
from opencompass.datasets.cmphysbench import CMPhysBenchEvaluator

cmphysbench_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='ground_truth'
)

cmphysbench_datasets = []
cmphysbench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(role='HUMAN', prompt='You are a condensed matter physics expert. Please read the following question and provide a step-by-step solution using only the given symbols. Do not introduce any new symbols that are not provided in the problem statement. Your final answer must be presented as a readable LaTeX formula, enclosed in a \\boxed{{}} environment.\n{prompt}'),
            dict(role='BOT', prompt='{ground_truth}\n')
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
cmphysbench_eval_cfg = dict(
    evaluator=dict(type=CMPhysBenchEvaluator),
)

cmphysbench_datasets.append(
    dict(
        abbr='CMPhysBench-fix_prompt',
        type=CMPhysBenchDataset,
        path='weidawang/CMPhysBench',
        reader_cfg=cmphysbench_reader_cfg,
        infer_cfg=cmphysbench_infer_cfg,
        eval_cfg=cmphysbench_eval_cfg,
    )
)
