from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import SQuAD20Dataset, SQuAD20Evaluator

squad20_reader_cfg = dict(
    input_columns=['context', 'question'],
    output_column='answers')

squad20_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='{context}\nAccording to the above passage, answer the following question. If it is impossible to answer according to the passage, answer `impossible to answer`:\nQuestion: {question}'),
                dict(role='BOT', prompt='Answer:'),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=50))

squad20_eval_cfg = dict(
    evaluator=dict(type=SQuAD20Evaluator), pred_role='BOT')

squad20_datasets = [
    dict(
        type=SQuAD20Dataset,
        abbr='squad2.0',
        path='./data/SQuAD2.0/dev-v2.0.json',
        reader_cfg=squad20_reader_cfg,
        infer_cfg=squad20_infer_cfg,
        eval_cfg=squad20_eval_cfg)
]
