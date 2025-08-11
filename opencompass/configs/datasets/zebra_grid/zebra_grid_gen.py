from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import ZebraGridDataset
from opencompass.datasets.zebra_grid import ZebraGridEvaluator

zebra_grid_reader_cfg = dict(
    input_columns=['puzzle'],
    output_column='solution',  # Use solution field as placeholder for OpenCompass framework
    train_split='test',  # Use test split for both since only test split exists
    test_split='test'
)

zebra_grid_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='{puzzle}'  # Simple template since prompt is already formatted in dataset
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

zebra_grid_eval_cfg = dict(
    evaluator=dict(type=ZebraGridEvaluator),
    pred_role='BOT'
)

zebra_grid_datasets = [
    dict(
        abbr='zebra_grid',
        type=ZebraGridDataset,
        path='',  # No path needed for HuggingFace dataset
        reader_cfg=zebra_grid_reader_cfg,
        infer_cfg=zebra_grid_infer_cfg,
        eval_cfg=zebra_grid_eval_cfg,
    )
] 