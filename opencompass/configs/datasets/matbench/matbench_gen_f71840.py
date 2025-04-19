from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets.matbench.matbench import MatbenchDataset, MatbenchEvaluator_regression, MatbenchEvaluator_classification



matbench_reader_cfg = dict(
    input_columns=['problem'], output_column='answer')


matbench_tasks =  ['matbench_steels','matbench_expt_gap', 'matbench_expt_is_metal','matbench_glass']

matbench_datasets = []

for task in matbench_tasks:
    if task in ['matbench_expt_is_metal','matbench_glass']:
        matbench_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[dict(role='HUMAN', prompt=f'{{problem}} Please present your answer by yes or no, do not output anything else.')])),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer))

        matbench_eval_cfg = dict(
            evaluator=dict(type=MatbenchEvaluator_classification),
            pred_role='BOT')

    elif task in ['matbench_steels','matbench_expt_gap']:
        matbench_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[dict(role='HUMAN', prompt=f'{{problem}} Please present your answer by one float number, do not output anything else.')])),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer))


        matbench_eval_cfg = dict(
            evaluator=dict(type=MatbenchEvaluator_regression),
            pred_role='BOT')


    matbench_datasets.append(
        dict(
            type=MatbenchDataset,
            path=f'opencompass/Matbench',
            task=task,
            abbr=task,
            reader_cfg=matbench_reader_cfg,
            infer_cfg=matbench_infer_cfg,
            eval_cfg=matbench_eval_cfg))

