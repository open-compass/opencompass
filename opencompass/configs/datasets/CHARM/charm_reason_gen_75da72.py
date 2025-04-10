import os
from mmengine.config import read_base

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import CharmDataset, charm_reason_postprocess, CharmReasonEvaluator

with read_base():
    from .charm_reason_settings import charm_tasks, settings


charm_reason_datasets = []

for _cot, _cot_prefix, dataset_path, fewshot_example_path, prompt_template in settings:
    for _task in charm_tasks:
        _fewshot_example_file = os.path.join(fewshot_example_path, f'{_task}_{_cot}.txt')
        with open(_fewshot_example_file, 'r') as f:
            _hint = f.read()

        charm_reason_reader_cfg = dict(input_columns=['input'], output_column='target')

        charm_reason_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[dict(role='HUMAN', prompt=prompt_template.format(_hint=_hint) + _cot_prefix)]),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer, max_out_len=512),
        )

        charm_reason_eval_cfg = dict(
            evaluator=dict(type=CharmReasonEvaluator),
            pred_role='BOT',
            pred_postprocessor=dict(type=charm_reason_postprocess),
            dataset_postprocessor=dict(type=charm_reason_postprocess),
        )

        charm_reason_datasets.append(
            dict(
                type=CharmDataset,
                path=dataset_path,
                name=_task,
                abbr='charm-reason-' + _task + '_' + _cot,
                reader_cfg=charm_reason_reader_cfg,
                infer_cfg=charm_reason_infer_cfg.copy(),
                eval_cfg=charm_reason_eval_cfg.copy(),
            )
        )
