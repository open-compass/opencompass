from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import MATHDataset, MATHEvaluator, math_postprocess_v2, GaoKaoMATHEvaluator
from opencompass.utils.model_postprocessors import naive_model_postprocess, xfinder_postprocess
from opencompass.utils.postprocessors.naive import MATH_NAVIE_PROMPT_TEMPLATE

# ----------------------------- Eval Parameters -----------------------------
## Postprocess function
post_func = 're' # 're', 'xfinder_model', 'naive_model'

## Evalute function
eval_func = 'naive_model' # 're', 'naive_model'

## Model api url
xfinder_url = 'http://0.0.0.0:23333/v1' # for 'xFinder-qwen1505' if post_func is 'xfinder_model'
naive_model_name = 'Qwen/Qwen2.5-72B-Instruct' # replace with your model name
naive_model_url = ['http://22.8.6.22:23333/v1', 'http://22.8.67.84:23333/v1', 'http://22.8.72.81:23333/v1', 'http://22.9.42.143:23333/v1'] # Multi-apis for accerlation

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
    inferencer=dict(type=GenInferencer, max_out_len=1024),
)

if post_func == 're':
    pred_postprocessor = dict(type=math_postprocess_v2)
elif post_func == 'xfinder_model':
    pred_postprocessor = dict(
        type=xfinder_postprocess,
        question_type='math',
        model_name='xFinder-qwen1505',
        num_processes=128,
        api_url=xfinder_url,
    )
elif post_func == 'naive_model':
    pred_postprocessor = dict(
        type=naive_model_postprocess,
        custom_instruction=MATH_NAVIE_PROMPT_TEMPLATE,
        model_name=naive_model_name,
        num_processes=64,
        api_url=naive_model_url,
    )

if eval_func == 're':
    evaluator = dict(type=MATHEvaluator, version='v2')
elif eval_func == 'naive_model':
    evaluator = dict(
        type=GaoKaoMATHEvaluator,
        model_name=naive_model_name,
        url=naive_model_url,
    )

math_eval_cfg = dict(
    evaluator=evaluator, pred_postprocessor=pred_postprocessor,
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
