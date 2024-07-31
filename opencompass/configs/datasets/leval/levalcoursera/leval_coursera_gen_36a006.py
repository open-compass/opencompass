from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import EMEvaluator, RougeEvaluator, SquadEvaluator, AccEvaluator
from opencompass.datasets.leval import LEvalCourseraDataset
from opencompass.utils.text_postprocessors import first_capital_postprocess, first_capital_postprocess_multi

LEval_coursera_reader_cfg = dict(
    input_columns=['context', 'question'],
    output_column='answer',
    train_split='test',
    test_split='test'
)

LEval_coursera_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(role='SYSTEM', fallback_role='HUMAN', prompt='Now you are given a very long document. Please follow the instruction based on this document. For multi-choice questions, there could be a single correct option or multiple correct options. Please only provide the letter corresponding to the answer (like A or AB) when answering.'),
            ],
            round=[
                dict(role='HUMAN', prompt='Document is as follows.\n{context}\nQuestion:{question}\nAnswer:'),
                dict(role='BOT', prompt=''),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=10)
)

LEval_coursera_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_postprocessor=dict(type=first_capital_postprocess_multi),
    pred_role='BOT'
)

LEval_coursera_datasets = [
    dict(
        type=LEvalCourseraDataset,
        abbr='LEval_coursera',
        path='L4NLP/LEval',
        name='coursera',
        reader_cfg=LEval_coursera_reader_cfg,
        infer_cfg=LEval_coursera_infer_cfg,
        eval_cfg=LEval_coursera_eval_cfg)
]
