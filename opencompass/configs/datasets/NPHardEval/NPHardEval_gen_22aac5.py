from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.NPHardEval import (
    HardGCPDataset, HardGCPEvaluator,
    Hard_TSP_Dataset, Hard_TSP_Evaluator,
    Hard_MSP_Dataset, Hard_MSP_Evaluator,
    CMP_GCP_D_Dataset, CMP_GCP_D_Evaluator,
    CMP_TSP_D_Dataset, CMP_TSP_D_Evaluator,
    CMP_KSP_Dataset, CMP_KSP_Evaluator,
    P_BSP_Dataset, P_BSP_Evaluator,
    P_EDP_Dataset, P_EDP_Evaluator,
    P_SPP_Dataset, P_SPP_Evaluator,
)

NPHardEval_tasks = [
    ['hard_GCP', 'GCP', HardGCPDataset, HardGCPEvaluator],
    ['hard_TSP', 'TSP', Hard_TSP_Dataset, Hard_TSP_Evaluator],
    ['hard_MSP', 'MSP', Hard_MSP_Dataset, Hard_MSP_Evaluator],
    ['cmp_GCP_D', 'GCP_Decision', CMP_GCP_D_Dataset, CMP_GCP_D_Evaluator],
    ['cmp_TSP_D', 'TSP_Decision', CMP_TSP_D_Dataset, CMP_TSP_D_Evaluator],
    ['cmp_KSP', 'KSP', CMP_KSP_Dataset, CMP_KSP_Evaluator],
    ['p_BSP', 'BSP', P_BSP_Dataset, P_BSP_Evaluator],
    ['p_EDP', 'EDP', P_EDP_Dataset, P_EDP_Evaluator],
    ['p_SPP', 'SPP', P_SPP_Dataset, P_SPP_Evaluator],
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
