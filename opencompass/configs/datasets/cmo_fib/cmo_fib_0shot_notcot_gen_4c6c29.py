from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import CMOFibDataset, MATHEvaluator, math_postprocess_v2


cmo_fib_reader_cfg = dict(
    input_columns=['question'], 
    output_column='answer'
)


cmo_fib_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='{question}\n你需要讲最终答案写入\\boxed{}.'),
            ],
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=2048)
)

cmo_fib_eval_cfg = dict(
    evaluator=dict(type=MATHEvaluator, version='v2'), pred_postprocessor=dict(type=math_postprocess_v2)
)

cmo_fib_datasets = [
    dict(
        abbr='cmo_fib',
        type=CMOFibDataset,
        path='opencompass/cmo_fib',
        reader_cfg=cmo_fib_reader_cfg,
        infer_cfg=cmo_fib_infer_cfg,
        eval_cfg=cmo_fib_eval_cfg
    )
]