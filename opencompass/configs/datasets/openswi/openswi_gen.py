from opencompass.openicl.icl_prompt_template import PromptTemplate
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
            type=PromptTemplate,
            template=dict(round=[
                dict(role='HUMAN',
                     prompt='{prompt}\nPlease think step by step, and put your final answer using python list.'),
                dict(role='BOT', prompt='{ground_truth}\n')
            ]),
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
