from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import EMEvaluator, RougeEvaluator, SquadEvaluator, AccEvaluator
from opencompass.datasets.leval import LEvalTopicRetrievalDataset, LEvalEMEvaluator
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
            begin=[
                dict(role='SYSTEM', fallback_role='HUMAN', prompt='Below is a record of our previous conversation on many different topics. You are the ASSISTANT, and I am the USER. At the beginning of each topic, the USER will say \'I would like to discuss the topic of <TOPIC>\'. Memorize each <TOPIC>. At the end of the record, I will ask you to retrieve the first/second/third topic names. Now the record start.'),
            ],
            round=[
                dict(role='HUMAN', prompt='Document is as follows.\n{context}\nQuestion:{question}\nAnswer:'),
                dict(role='BOT', prompt=''),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=30)
)

LEval_tr_eval_cfg = dict(
    evaluator=dict(type=LEvalEMEvaluator),
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
