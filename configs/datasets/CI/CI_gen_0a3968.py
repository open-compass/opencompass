from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer

from opencompass.datasets import CIDataset, CIEvaluator

cidataset_reader_cfg = dict(
    input_columns=["questions"],
    output_column="references",
    train_split='test',
    test_split='test')

cidataset_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="""{questions}""",
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)


libs = ['Pandas', 'Matplotlib', 'Opencv', 'SciPy', 'Seaborn', 'PyTorch']
cidataset_eval_cfg = {
    lib: dict(
        evaluator=dict(
            type=CIEvaluator,
            output_dir=f'output_data/cidataset/{lib}'),
        pred_role="BOT",
    )
    for lib in libs
}

cidataset_datasets = [
    dict(
        abbr=f"cidataset_{lib}",
        type=CIDataset,
        path=f"backup_data/cidataset/{lib}",
        reader_cfg=cidataset_reader_cfg,
        infer_cfg=cidataset_infer_cfg,
        eval_cfg=cidataset_eval_cfg[lib],
    ) for lib in libs
]
