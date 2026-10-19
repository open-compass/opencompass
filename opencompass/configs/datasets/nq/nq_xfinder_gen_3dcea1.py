from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import NaturalQuestionDataset, NQEvaluator
from opencompass.utils.model_postprocessors import xfinder_postprocess

nq_reader_cfg = dict(
    input_columns=['question'], output_column='answer', train_split='test')

nq_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='Question: {question}?\nAnswer: '),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

nq_eval_cfg = dict(
    evaluator=dict(type=NQEvaluator), pred_role='BOT',
    model_postprocessor=dict(
        type=xfinder_postprocess,
        question_type='short_text',
        xfinder_model_name='xFinder-qwen1505',
        xfiner_api_url='http://0.0.0.0:23333/v1,http://0.0.0.0:23334/v1')
    )

nq_datasets = [
    dict(
        type=NaturalQuestionDataset,
        abbr='nq',
        path='opencompass/natural_question',
        reader_cfg=nq_reader_cfg,
        infer_cfg=nq_infer_cfg,
        eval_cfg=nq_eval_cfg)
]
