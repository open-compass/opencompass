from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import AgentInferencer

from opencompass.datasets import CIDataset, CIEvaluator

ci_reader_cfg = dict(
    input_columns=["questions"],
    output_column="references",
    train_split='test',
    test_split='test')

ci_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="""{questions}""",
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=AgentInferencer),
)


libs = ['Pandas', 'Matplotlib', 'Opencv', 'SciPy', 'Seaborn', 'PyTorch']
ci_eval_cfg = {
    lib: dict(
        evaluator=dict(
            type=CIEvaluator,
            output_dir=f'output_data/cidataset/{lib}'),
        pred_role="BOT",
    )
    for lib in libs
}

ci_datasets = [
    dict(
        abbr=f"ci_{lib}",
        type=CIDataset,
        path=f"./data/cidataset/{lib}",
        reader_cfg=ci_reader_cfg,
        infer_cfg=ci_infer_cfg,
        eval_cfg=ci_eval_cfg[lib],
    ) for lib in libs
]
