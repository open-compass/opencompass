from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.NPHardEval import (
    hard_GCP_Dataset, hard_GCP_Evaluator,
    hard_TSP_Dataset, hard_TSP_Evaluator,
    hard_MSP_Dataset, hard_MSP_Evaluator,
    cmp_GCP_D_Dataset, cmp_GCP_D_Evaluator,
    cmp_TSP_D_Dataset, cmp_TSP_D_Evaluator,
    cmp_KSP_Dataset, cmp_KSP_Evaluator,
    p_BSP_Dataset, p_BSP_Evaluator,
    p_EDP_Dataset, p_EDP_Evaluator,
    p_SPP_Dataset, p_SPP_Evaluator,
)

NPHardEval_tasks = [
    ['hard_GCP', 'GCP', hard_GCP_Dataset, hard_GCP_Evaluator],
    ['hard_TSP', 'TSP', hard_TSP_Dataset, hard_TSP_Evaluator],
    ['hard_MSP', 'MSP', hard_MSP_Dataset, hard_MSP_Evaluator],
    ['cmp_GCP_D', 'GCP_Decision', cmp_GCP_D_Dataset, cmp_GCP_D_Evaluator],
    ['cmp_TSP_D', 'TSP_Decision', cmp_TSP_D_Dataset, cmp_TSP_D_Evaluator],
    ['cmp_KSP', 'KSP', cmp_KSP_Dataset, cmp_KSP_Evaluator],
    ['p_BSP', 'BSP', p_BSP_Dataset, p_BSP_Evaluator],
    ['p_EDP', 'EDP', p_EDP_Dataset, p_EDP_Evaluator],
    ['p_SPP', 'SPP', p_SPP_Dataset, p_SPP_Evaluator],
]

NPHardEval_datasets = []
for name, path_name, dataset, evaluator in NPHardEval_tasks:
    NPHardEval_reader_cfg = dict(input_columns=['prompt', 'level'], output_column='q')

    NPHardEval_infer_cfg = dict(
        ice_template=dict(
            type=PromptTemplate,
            template=dict(
                begin='</E>',
                round=[
                    dict(role='HUMAN', prompt='</E>{prompt}'),
                    dict(role='BOT', prompt=''),
                ],
            ),
            ice_token='</E>',
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )

    NPHardEval_eval_cfg = dict(evaluator=dict(type=evaluator), pred_role='BOT')

    NPHardEval_datasets.append(
        dict(
            type=dataset,
            abbr=name,
            path=f'./data/NPHardEval/{path_name}/',
            reader_cfg=NPHardEval_reader_cfg,
            infer_cfg=NPHardEval_infer_cfg,
            eval_cfg=NPHardEval_eval_cfg,
        )
    )
