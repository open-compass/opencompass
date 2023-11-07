from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import AgentInferencer

from opencompass.datasets import CIBenchDataset, CIBenchEvaluator

cibench_reader_cfg = dict(
    input_columns=["questions"],
    output_column="references",
    train_split='test',
    test_split='test')

cibench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="""{questions}""",
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=AgentInferencer),
)


libs = ['Pandas', 'Matplotlib', 'Opencv', 'SciPy', 'Seaborn', 'PyTorch']
cibench_eval_cfg = {
    lib: dict(
        evaluator=dict(
            type=CIBenchEvaluator,
            output_dir=f'output_data/cibench/{lib}'),
        pred_role="BOT",
    )
    for lib in libs
}

cibench_datasets = [
    dict(
        abbr=f"cibench_{lib}",
        type=CIBenchDataset,
        path=f"./data/cibench/{lib}",
        reader_cfg=cibench_reader_cfg,
        infer_cfg=cibench_infer_cfg,
        eval_cfg=cibench_eval_cfg[lib],
    ) for lib in libs
]
