from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import HumanevalevalProDataset, HumanevalProEvaluator, humaneval_postprocess_v2


PROMPT_WRAPPER = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.
Write a solution of python file to the following problems, the solution of the second problem requires single or multiple calls to the first solution.
```python
{raw_problem}
{new_problem}
```
Please put the two solutions within the Python code block provided below, and make sure that the block contains no other unrelated content:
```python
```
"""


humanevalpro_reader_cfg = dict(
    input_columns=['raw_problem', 'new_problem'], output_column='test_code')

humanevalpro_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=PROMPT_WRAPPER),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

humanevalpro_eval_cfg = dict(
    evaluator=dict(type=HumanevalProEvaluator,
                   ip_address='https://opencompass-multiple-evaluator.hf.space')
)

humanevalpro_datasets = [
    dict(
        abbr='humaneval_pro',
        type=HumanevalevalProDataset,
        path='opencompass/humaneval_pro',
        reader_cfg=humanevalpro_reader_cfg,
        infer_cfg=humanevalpro_infer_cfg,
        eval_cfg=humanevalpro_eval_cfg,
        n=5,
        k=3)
]