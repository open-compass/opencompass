from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import EMEvaluator, RougeEvaluator, SquadEvaluator
from opencompass.datasets.leval import LEvalGPTEvaluator, LEvalPatentSummDataset

LEval_patent_summ_reader_cfg = dict(
    input_columns=['context', 'question', 'length'],
    output_column='answer',
    train_split='test',
    test_split='test'
)

LEval_patent_summ_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(role='SYSTEM', fallback_role='HUMAN', prompt='Now you are given a very long document. Please follow the instruction after this document. These instructions may include summarizing a document, answering questions based on the document, or writing a required paragraph.'),
            ],
            round=[
                dict(role='HUMAN', prompt='Document is as follows. {context}\nInstruction: {question}\nAnswer this question with {length} words.'),
                dict(role='BOT', prompt=''),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512)
)

LEval_patent_summ_eval_cfg = dict(
    evaluator=dict(type=LEvalGPTEvaluator),
    pred_role='BOT'
)

LEval_patent_summ_datasets = [
    dict(
        type=LEvalPatentSummDataset,
        abbr='LEval_patent_summ',
        path='L4NLP/LEval',
        name='patent_summ',
        reader_cfg=LEval_patent_summ_reader_cfg,
        infer_cfg=LEval_patent_summ_infer_cfg,
        eval_cfg=LEval_patent_summ_eval_cfg)
]
