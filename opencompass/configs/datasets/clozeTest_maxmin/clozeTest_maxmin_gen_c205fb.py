from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import MaxminDataset
from opencompass.utils.text_postprocessors import first_capital_postprocess


maxmin_reader_cfg = dict(
    input_columns=['nl_tokens', 'pl_tokens'],
    output_column='answer',
)

maxmin_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt="Code:{pl_tokens}\nThe aim of the code: {nl_tokens}\nQuestion: Please tell me what \"<mask>\" in the code should be replaced with and you must response to me only A or B.\nA. max\nB. min\nAnswer:"),
                dict(role='BOT', prompt='{answer}'),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

maxmin_eval_cfg = dict(evaluator=dict(type=AccEvaluator),
                        pred_role='BOT',
                        pred_postprocessor=dict(type=first_capital_postprocess))

maxmin_datasets = [
    dict(
        type=MaxminDataset,
        abbr=f'maxmin',
        test_path='opencompass/clozeTest_maxmin',
        answer_path='opencompass/clozeTest_maxmin_answers',
        reader_cfg=maxmin_reader_cfg,
        infer_cfg=maxmin_infer_cfg,
        eval_cfg=maxmin_eval_cfg,
    )
]
