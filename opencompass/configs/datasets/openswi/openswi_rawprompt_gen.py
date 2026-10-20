from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.OpenSWI import OpenSWIDataset
from opencompass.datasets.OpenSWI import OpenSWIMSEEvaluator

openswi_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='ground_truth'
)

name_set = [
    'shallow',
    'deep',
]

openswi_datasets = []
for _name in name_set:
    openswi_infer_cfg = dict(
        prompt_template=dict(
            type=RawPromptTemplate,
            messages=[
                {'role': 'system', 'content': 'You are a professional geophysical inversion expert, proficient in utilizing surface wave dispersion data to infer subsurface S-wave velocity structures. Based on the provided surface wave dispersion data, perform nonlinear inversion to obtain an S-wave velocity (Vs) sequence at specified depth points.'},
                {'role': 'user', 'content': '{prompt}\nPlease think step by step, and put your final answer using python list.'},
            ],
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )
    openswi_eval_cfg = dict(
        evaluator=dict(type=OpenSWIMSEEvaluator),
    )

    openswi_datasets.append(
        dict(
            abbr=f'OpenSWI-{_name}-1k',
            type=OpenSWIDataset,
            name=_name,
            path='opencompass/openswi',
            reader_cfg=openswi_reader_cfg,
            infer_cfg=openswi_infer_cfg,
            eval_cfg=openswi_eval_cfg,
        )
    )

del _name
