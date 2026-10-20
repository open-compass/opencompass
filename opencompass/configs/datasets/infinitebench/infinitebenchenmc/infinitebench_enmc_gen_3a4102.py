from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.utils.text_postprocessors import first_option_postprocess
from opencompass.datasets import InfiniteBenchenmcDataset

InfiniteBench_enmc_reader_cfg = dict(
    input_columns=['context', 'question', 'option_A', 'option_B', 'option_C', 'option_D'],
    output_column='answer',

)

InfiniteBench_enmc_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(role='SYSTEM', fallback_role='HUMAN', prompt='You are a helpful assistant.'),
            ],
            round=[
                dict(role='HUMAN', prompt='Read the book and answer the question.\n\n{context}\n\nQuestion: {question}\n\nOnly one of the following options is correct, tell me the answer using one single letter (A, B, C, or D). Don\'t say anything else.\nA. {option_A}\nB. {option_B}\nC. {option_C}\nD. {option_D}'),
                dict(role='BOT', prompt=''),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=40)
)

InfiniteBench_enmc_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
    pred_role='BOT'
)

InfiniteBench_enmc_datasets = [
    dict(
        type=InfiniteBenchenmcDataset,
        abbr='InfiniteBench_enmc',
        path='./data/InfiniteBench/longbook_choice_eng.jsonl',
        reader_cfg=InfiniteBench_enmc_reader_cfg,
        infer_cfg=InfiniteBench_enmc_infer_cfg,
        eval_cfg=InfiniteBench_enmc_eval_cfg)
]
