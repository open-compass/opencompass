from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import MATHDataset, GaoKaoMATHEvaluator
from opencompass.utils.model_postprocessors import naive_model_postprocess
from opencompass.utils.postprocessors.naive import MATH_NAVIE_PROMPT_TEMPLATE

# ----------------------------- Eval Parameters -----------------------------
## Postprocess info

naive_model_name = 'dlc_model' # replace with your model name
naive_model_url = ['http://0.0.0.0:23333/v1'] # Multi-apis for accerlation

# ----------------------------- Detailed Config -----------------------------

math_reader_cfg = dict(input_columns=['problem'], output_column='solution')

math_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='{problem}\nPlease reason step by step, and put your final answer within \\boxed{}.'),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=2048),
)

pred_postprocessor = dict(
    type=naive_model_postprocess,
    custom_instruction=MATH_NAVIE_PROMPT_TEMPLATE,
    model_name=naive_model_name,
    num_processes=128,
    api_url=naive_model_url,
)

evaluator = dict(
    type=GaoKaoMATHEvaluator,
    model_name=naive_model_name,
    url=naive_model_url,
)

math_eval_cfg = dict(
    evaluator=evaluator, model_postprocessor=pred_postprocessor,
)

math_datasets = [
    dict(
        type=MATHDataset,
        abbr='math',
        path='opencompass/math',
        reader_cfg=math_reader_cfg,
        infer_cfg=math_infer_cfg,
        eval_cfg=math_eval_cfg,
    )
]
