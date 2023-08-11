from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import EMEvaluator, RougeEvaluator, SquadEvaluator, AccEvaluator
from opencompass.datasets import LEvalTopicRetrievalDataset
from opencompass.utils.text_postprocessors import first_capital_postprocess, first_capital_postprocess_multi, general_postprocess

LEval_tr_reader_cfg = dict(
    input_columns=['context', 'question'],
    output_column='answer',
    train_split='test',
    test_split='test'
)

LEval_tr_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='{context}\nQuestion: {question}\nAnswer:'),
                dict(role='BOT', prompt=''),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=30)
)

LEval_tr_eval_cfg = dict(
    evaluator=dict(type=EMEvaluator), 
    pred_postprocessor=dict(type=general_postprocess),
    pred_role='BOT'
)

LEval_tr_datasets = [
    dict(
        type=LEvalTopicRetrievalDataset,
        abbr='LEval_topic_retrieval',
        path='L4NLP/LEval',
        name='topic_retrieval_longchat',
        reader_cfg=LEval_tr_reader_cfg,
        infer_cfg=LEval_tr_infer_cfg,
        eval_cfg=LEval_tr_eval_cfg)
]
