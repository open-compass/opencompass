import os
from mmengine.config import read_base

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import CharmDataset, CharmMemoryEvaluator, LMEvaluator

with read_base():
    from .charm_memory_settings import charm_memory_tasks, judge_system_prompts, dataset_path

charm_memory_datasets = []

for _task in charm_memory_tasks:

    charm_memory_reader_cfg = dict(input_columns=['input'],
                                   output_column='target')

    charm_memory_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(round=[
                dict(role='HUMAN', prompt='请尽可能简短地回答下述问题。\n问题：{input}\n答：')
            ]),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer, max_out_len=512),
    )

    if _task == 'Chinese_Movie_and_Music_Recommendation':
        charm_memory_eval_cfg = dict(
            evaluator=dict(type=CharmMemoryEvaluator),
            pred_role='BOT',
        )
    else:
        judge_system_prompt = judge_system_prompts[_task]
        charm_memory_eval_cfg = dict(
            evaluator=dict(
                type=LMEvaluator,
                prompt_template=dict(
                    type=PromptTemplate,
                    template=dict(round=[
                        dict(
                            role='HUMAN',
                            prompt=judge_system_prompt +
                            "\n\n[Question]\n{input}\n[The Start of Reference Answer]\n{target}\n[The End of Reference Answer]\n\n[The Start of Assistant's Answer]\n{prediction}\n[The End of Assistant's Answer]"  # noqa
                        ),
                    ]),
                ),
            ),
            pred_role='BOT',
        )

    charm_memory_datasets.append(
        dict(
            type=CharmDataset,
            path=dataset_path,
            name=_task,
            abbr='charm-memory-' + _task,
            reader_cfg=charm_memory_reader_cfg,
            infer_cfg=charm_memory_infer_cfg.copy(),
            eval_cfg=charm_memory_eval_cfg.copy(),
        ))
