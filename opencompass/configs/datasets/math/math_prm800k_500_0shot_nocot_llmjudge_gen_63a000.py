from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import MATHDataset, MATHEvaluator, math_postprocess_v2, normalize_final_answer
from opencompass.datasets import MATHDataset, MATHEvaluator, math_postprocess_v2, GaoKaoMATHEvaluator
# from opencompass.utils.model_postprocessors import naive_model_postprocess, xfinder_postprocess
from opencompass.utils.postprocessors.naive import MATH_NAVIE_PROMPT_TEMPLATE

# ----------------------------- Eval Parameters -----------------------------
## Postprocess function
post_func = 're' # 're', 'xfinder_model', 'naive_model'

## Evalute function
eval_func = 'naive_model' # 're', 'naive_model'


## Model api url
# xfinder_url = 'http://0.0.0.0:23333/v1' # for 'xFinder-qwen1505' if post_func is 'xfinder_model'
# naive_model_name = 'Qwen/Qwen2.5-72B-Instruct' # replace with your model name
naive_model_name = 'dlc_model'
# naive_model_url = [
#     'http://172.30.56.38:23001/v1',
# ] # Multi-apis for accerlation
naive_model_url = [
    "http://172.30.56.38:23001/v1",
    "http://172.30.8.4:23003/v1",
    "http://172.30.8.14:23002/v1",
    "http://172.30.48.80:23004/v1",
    "http://172.30.56.132:23005/v1",
    "http://172.30.16.115:23006/v1",
    "http://172.30.48.82:23007/v1",
    "http://172.30.24.53:23008/v1",
    "http://172.30.56.141:23009/v1",
    "http://172.30.8.35:23010/v1",
    "http://172.30.48.85:23011/v1",
    "http://172.30.16.116:23012/v1"
]
# ----------------------------- Detailed Config -----------------------------

math_reader_cfg = dict(input_columns=['problem'], output_column='solution')

math_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='{problem}\nRemember to put your final answer within \\boxed{}.'),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=8192),
)


if post_func == 're':
    pred_postprocessor = dict(type=math_postprocess_v2)


if eval_func == 're':
    evaluator = dict(type=MATHEvaluator, version='v2')
elif eval_func == 'naive_model':
    evaluator = dict(
        type=GaoKaoMATHEvaluator,
        judge_model_name=naive_model_name,
        url=naive_model_url,
    )

# postprocess v2
math_eval_cfg = dict(
    evaluator=evaluator,
    pred_postprocessor=pred_postprocessor,
)

math_datasets = [
    dict(
        type=MATHDataset,
        abbr='math_prm800k_500-llmjudge',
        path='opencompass/math',
        file_name = 'test_prm800k_500.json',
        reader_cfg=math_reader_cfg,
        infer_cfg=math_infer_cfg,
        eval_cfg=math_eval_cfg,
    )
]
