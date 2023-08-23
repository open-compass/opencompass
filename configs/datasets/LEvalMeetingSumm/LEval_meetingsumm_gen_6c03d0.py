from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import EMEvaluator, RougeEvaluator, SquadEvaluator
from opencompass.datasets import LEvalMeetingSummDataset

LEval_meetingsumm_reader_cfg = dict(
    input_columns=['context', 'question'],
    output_column='answer',
    train_split='test',
    test_split='test'
)

LEval_meetingsumm_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='{context}\nQuestion: {question}\nAnswer:'),
                dict(role='BOT', prompt=''),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512)
)

LEval_meetingsumm_eval_cfg = dict(
    evaluator=dict(type=RougeEvaluator), 
    pred_role='BOT'
)

LEval_meetingsumm_datasets = [
    dict(
        type=LEvalMeetingSummDataset,
        abbr='LEval_meeting_summ',
        path='L4NLP/LEval',
        name='meeting_summ',
        reader_cfg=LEval_meetingsumm_reader_cfg,
        infer_cfg=LEval_meetingsumm_infer_cfg,
        eval_cfg=LEval_meetingsumm_eval_cfg)
]
