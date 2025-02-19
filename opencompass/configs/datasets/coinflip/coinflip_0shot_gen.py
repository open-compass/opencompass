from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import CoinFlipDataset, coinflip_pred_postprocess

coinflip_reader_cfg = dict(
    input_columns=['question'],
    output_column='answer',
    train_split='test',
    test_split='test'
)

coinflip_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='Question: {question}\nPlease respond `yes` or `no` directly without any additional explanations.\nAnswer:'),
            ]
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512),
)

coinflip_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_postprocessor=dict(type=coinflip_pred_postprocess),
)

coinflip_datasets = [
    dict(
        abbr='coinflip',
        type=CoinFlipDataset,
        path='coin_flip',
        reader_cfg=coinflip_reader_cfg,
        infer_cfg=coinflip_infer_cfg,
        eval_cfg=coinflip_eval_cfg
    )
]