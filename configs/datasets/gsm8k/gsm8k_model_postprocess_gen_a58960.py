from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GSM8KDataset, gsm8k_dataset_postprocess
from opencompass.datasets import MATHEvaluator, math_postprocess_v2
from opencompass.utils.model_postprocessors import navie_model_postprocess
from opencompass.utils.postprocessors.naive import MATH_NAVIE_PROMPT_TEMPLATE

gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

gsm8k_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='{question}\nPlease reason step by step, and put your final answer within \\boxed{}.'),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512),
)

# # You can write your own postprocess prompt like:
# GSM8K_NAVIE_PROMPT_TEMPLATE = """
# There is a detailed explanation of the final answer you should extract:
# 1. ...
# 2. ...
# ...
# """

gsm8k_eval_cfg = dict(
    evaluator=dict(type=MATHEvaluator, version='v2'),
    pred_postprocessor=dict(type=math_postprocess_v2),
    dataset_postprocessor=dict(type=gsm8k_dataset_postprocess),
    model_postprocessor=dict(
        type=navie_model_postprocess,
        custom_instruction=MATH_NAVIE_PROMPT_TEMPLATE,
        model_name='',
        api_url='http://0.0.0.0:23333/v1,http://0.0.0.0:23334/v1')
    )

gsm8k_datasets = [
    dict(
        abbr='gsm8k',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg,
    )
]
