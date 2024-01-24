from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.NPHardEval import cmp_GCP_D_Dataset, cmp_GCP_D_Evaluator
from opencompass.datasets.NPHardEval import cmp_KSP_Dataset, cmp_KSP_Evaluator
from opencompass.datasets.NPHardEval import cmp_TSP_D_Dataset, cmp_TSP_D_Evaluator
from opencompass.datasets.NPHardEval import hard_GCP_Dataset, hard_GCP_Evaluator
from opencompass.datasets.NPHardEval import hard_MSP_Dataset, hard_MSP_Evaluator
from opencompass.datasets.NPHardEval import hard_TSP_Dataset, hard_TSP_Evaluator
from opencompass.datasets.NPHardEval import p_BSP_Dataset, p_BSP_Evaluator
from opencompass.datasets.NPHardEval import p_EDP_Dataset, p_EDP_Evaluator
from opencompass.datasets.NPHardEval import p_SPP_Dataset, p_SPP_Evaluator

cmp_GCP_D_reader_cfg = dict(
    input_columns=['prompt', 'level'], output_column='q')
cmp_KSP_reader_cfg = dict(
    input_columns=['prompt', 'level'], output_column='q')
cmp_TSP_D_reader_cfg = dict(
    input_columns=['prompt', 'level'], output_column='q')
hard_GCP_reader_cfg = dict(
    input_columns=['prompt', 'level'], output_column='q')
hard_MSP_reader_cfg = dict(
    input_columns=['prompt', 'level'], output_column='q')
hard_TSP_reader_cfg = dict(
    input_columns=['prompt', 'level'], output_column='q')
p_BSP_reader_cfg = dict(
    input_columns=['prompt', 'level'], output_column='q')
p_EDP_reader_cfg = dict(
    input_columns=['prompt', 'level'], output_column='q')
p_SPP_reader_cfg = dict(
    input_columns=['prompt', 'level'], output_column='q')

NPHardEval_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(
            begin="</E>",
            round=[
                dict(
                    role="HUMAN",
                    prompt="</E>{prompt}"
                ),
                dict(role="BOT", prompt=""),
            ]),
        ice_token="</E>",
        ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

cmp_GCP_D_eval_cfg = dict(evaluator=dict(type=cmp_GCP_D_Evaluator), pred_role="BOT")
cmp_KSP_eval_cfg = dict(evaluator=dict(type=cmp_KSP_Evaluator), pred_role="BOT")
cmp_TSP_D_eval_cfg = dict(evaluator=dict(type=cmp_TSP_D_Evaluator), pred_role="BOT")
hard_GCP_eval_cfg = dict(evaluator=dict(type=hard_GCP_Evaluator), pred_role="BOT")
hard_MSP_eval_cfg = dict(evaluator=dict(type=hard_MSP_Evaluator), pred_role="BOT")
hard_TSP_eval_cfg = dict(evaluator=dict(type=hard_TSP_Evaluator), pred_role="BOT")
p_BSP_eval_cfg = dict(evaluator=dict(type=p_BSP_Evaluator), pred_role="BOT")
p_EDP_eval_cfg = dict(evaluator=dict(type=p_EDP_Evaluator), pred_role="BOT")
p_SPP_eval_cfg = dict(evaluator=dict(type=p_SPP_Evaluator), pred_role="BOT")

nphard_datasets = [
    dict(
        type=cmp_GCP_D_Dataset,
        abbr='cmp_GCP_D',
        path='./data/NPHardEval/Zeroshot/GCP_Decision/',
        reader_cfg=cmp_GCP_D_reader_cfg,
        infer_cfg=NPHardEval_infer_cfg,
        eval_cfg=cmp_GCP_D_eval_cfg),
    dict(
        type=cmp_KSP_Dataset,
        abbr='cmp_KSP',
        path='./data/NPHardEval/Zeroshot/KSP/',
        reader_cfg=cmp_KSP_reader_cfg,
        infer_cfg=NPHardEval_infer_cfg,
        eval_cfg=cmp_KSP_eval_cfg),
    dict(
        type=cmp_TSP_D_Dataset,
        abbr='cmp_TSP_D',
        path='./data/NPHardEval/Zeroshot/TSP_Decision/',
        reader_cfg=cmp_TSP_D_reader_cfg,
        infer_cfg=NPHardEval_infer_cfg,
        eval_cfg=cmp_TSP_D_eval_cfg),
    dict(
        type=hard_GCP_Dataset,
        abbr='hard_GCP',
        path='./data/NPHardEval/Zeroshot/GCP/',
        reader_cfg=hard_GCP_reader_cfg,
        infer_cfg=NPHardEval_infer_cfg,
        eval_cfg=hard_GCP_eval_cfg),
    dict(
        type=hard_MSP_Dataset,
        abbr='hard_MSP',
        path='./data/NPHardEval/Zeroshot/MSP/',
        reader_cfg=hard_MSP_reader_cfg,
        infer_cfg=NPHardEval_infer_cfg,
        eval_cfg=hard_MSP_eval_cfg),
    dict(
        type=hard_TSP_Dataset,
        abbr='hard_TSP',
        path='./data/NPHardEval/Zeroshot/TSP/',
        reader_cfg=hard_TSP_reader_cfg,
        infer_cfg=NPHardEval_infer_cfg,
        eval_cfg=hard_TSP_eval_cfg),
    dict(
        type=p_BSP_Dataset,
        abbr='p_BSP',
        path='./data/NPHardEval/Zeroshot/BSP/',
        reader_cfg=p_BSP_reader_cfg,
        infer_cfg=NPHardEval_infer_cfg,
        eval_cfg=p_BSP_eval_cfg),
    dict(
        type=p_EDP_Dataset,
        abbr='p_EDP',
        path='./data/NPHardEval/Zeroshot/EDP/',
        reader_cfg=p_EDP_reader_cfg,
        infer_cfg=NPHardEval_infer_cfg,
        eval_cfg=p_EDP_eval_cfg),
    dict(
        type=p_SPP_Dataset,
        abbr='p_SPP',
        path='./data/NPHardEval/Zeroshot/SPP/',
        reader_cfg=p_SPP_reader_cfg,
        infer_cfg=NPHardEval_infer_cfg,
        eval_cfg=p_SPP_eval_cfg)
]
