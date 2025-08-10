from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import EMEvaluator, RougeEvaluator, SquadEvaluator, AccEvaluator
from opencompass.datasets.leval import LEvalGSM100Dataset
from opencompass.utils.text_postprocessors import first_capital_postprocess, first_capital_postprocess_multi
from opencompass.registry import TEXT_POSTPROCESSORS
from opencompass.datasets import gsm100_dataset_postprocess, gsm100_postprocess

LEval_gsm100_reader_cfg = dict(
    input_columns=['context', 'question'],
    output_column='answer',
    train_split='test',
    test_split='test'
)

LEval_gsm100_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(role='SYSTEM', fallback_role='HUMAN', prompt='Given several question answer pairs, you need to follow a similar format to answer the last question. Make sure the response is end with The answer is _ . '),
            ],
            round=[
                dict(role='HUMAN', prompt='{context}\n\n{question}\n'),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512)
)


LEval_gsm100_eval_cfg = dict(evaluator=dict(type=AccEvaluator),
    pred_postprocessor=dict(type=gsm100_postprocess),
    dataset_postprocessor=dict(type=gsm100_dataset_postprocess)
)

LEval_gsm100_datasets = [
    dict(
        type=LEvalGSM100Dataset,
        abbr='LEval_gsm100',
        path='L4NLP/LEval',
        name='gsm100',
        reader_cfg=LEval_gsm100_reader_cfg,
        infer_cfg=LEval_gsm100_infer_cfg,
        eval_cfg=LEval_gsm100_eval_cfg)
]
