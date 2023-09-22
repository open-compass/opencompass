from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import AttackInferencer
from opencompass.datasets import SQuAD20Dataset, SQuAD20Evaluator

squad20_reader_cfg = dict(
    input_columns=['context', 'question'],
    output_column='answers')

original_prompt_list = [
    "Based on the given context, provide the best possible answer. If there's no answer available in the context, respond with 'unanswerable'.",
    "Identify the most relevant answer from the context. If it's not possible to find an answer, respond with 'unanswerable'.",
    "Find the correct answer in the context provided. If an answer cannot be found, please respond with 'unanswerable'.",
    "Please extract the most appropriate answer from the context. If an answer is not present, indicate 'unanswerable'.",
    "Using the context, determine the most suitable answer. If the context doesn't contain the answer, respond with 'unanswerable'.",
    "Locate the most accurate answer within the context. If the context doesn't provide an answer, respond with 'unanswerable'.",
    "Please derive the most fitting answer from the context. If there isn't an answer in the context, respond with 'unanswerable'.",
    "Discover the best answer based on the context. If the context doesn't include an answer, respond with 'unanswerable'.",
    "From the context, provide the most precise answer. If the answer is not in the context, respond with 'unanswerable'.",
    "Search the context for the most relevant answer. If the answer cannot be found, respond with 'unanswerable'.",
]

squad20_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='{adv_prompt} {context}'),
                dict(role='BOT', prompt='Answer:'),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=AttackInferencer, max_out_len=50,
        original_prompt_list=original_prompt_list,
        adv_key='adv_prompt',
        metric_key='score'))

squad20_eval_cfg = dict(
    evaluator=dict(type=SQuAD20Evaluator), pred_role='BOT')

squad20_datasets = [
    dict(
        type=SQuAD20Dataset,
        abbr='squad_v2',
        path='./data/SQuAD2.0/dev-v2.0.json',
        reader_cfg=squad20_reader_cfg,
        infer_cfg=squad20_infer_cfg,
        eval_cfg=squad20_eval_cfg)
]
