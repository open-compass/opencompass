import os
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import BBEHDataset, BBEHEvaluator, bbeh_mcq_postprocess, BBEHEvaluator_mcq

bbeh_reader_cfg = dict(input_columns=['input'], output_column='target')


bbeh_multiple_choice_sets = [
    'bbeh_boolean_expressions',
    'bbeh_disambiguation_qa',
    'bbeh_geometric_shapes',
    'bbeh_hyperbaton',
    'bbeh_movie_recommendation',
    'bbeh_nycc',
    'bbeh_shuffled_objects',
]

bbeh_free_form_sets = [
    'bbeh_boardgame_qa',
    'bbeh_buggy_tables',
    'bbeh_causal_understanding',
    'bbeh_dyck_languages',
    'bbeh_linguini',
    'bbeh_multistep_arithmetic',
    'bbeh_object_counting',
    'bbeh_object_properties',
    'bbeh_sarc_triples',
    'bbeh_spatial_reasoning',
    'bbeh_sportqa',
    'bbeh_temporal_sequence',
    'bbeh_time_arithmetic',
    'bbeh_web_of_lies',
    'bbeh_word_sorting',
    'bbeh_zebra_puzzles',
]

bbeh_datasets = []
for _name in bbeh_multiple_choice_sets:
    bbeh_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(round=[
                dict(
                    role='HUMAN',
                    prompt=
                    f"Think step by step, and when you provide the final answer, please use the prefix \"The answer is:\"without any modification, and provide the answer directly, with no formatting, no bolding, and no markup. For instance: \"The answer is: 42\" or \"The answer is: yes\". If the question is multiple choice with a single correct answer, the final answer must only be the letter corresponding to the correct answer. For example, \"The answer is: (a)\"\n\nQ: {{input}}\nA: "
                )
            ])),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer, max_out_len=8192))
    bbeh_eval_cfg = dict(
        evaluator=dict(type=BBEHEvaluator_mcq),
        pred_role='BOT',
        pred_postprocessor=dict(type=bbeh_mcq_postprocess),
        dataset_postprocessor=dict(type=bbeh_mcq_postprocess))

    bbeh_datasets.append(
        dict(
            type=BBEHDataset,
            path='opencompass/bbeh',
            name=_name,
            abbr=_name,
            reader_cfg=bbeh_reader_cfg,
            infer_cfg=bbeh_infer_cfg.copy(),
            eval_cfg=bbeh_eval_cfg.copy()))

for _name in bbeh_free_form_sets:
    bbeh_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(round=[
                dict(
                    role='HUMAN',
                    prompt=
                    f"Think step by step, and when you provide the final answer, please use the prefix \"The answer is:\"without any modification, and provide the answer directly, with no formatting, no bolding, and no markup. For instance: \"The answer is: 42\" or \"The answer is: yes\". If the question is multiple choice with a single correct answer, the final answer must only be the letter corresponding to the correct answer. For example, \"The answer is: (a)\"\n\nQ: {{input}}\nA: "
                )
            ])),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer, max_out_len=8192))
    bbeh_eval_cfg = dict(evaluator=dict(type=BBEHEvaluator), pred_role='BOT', pred_postprocessor=dict(type=bbeh_mcq_postprocess), dataset_postprocessor=dict(type=bbeh_mcq_postprocess))

    bbeh_datasets.append(
        dict(
            type=BBEHDataset,
            path='opencompass/bbeh',
            name=_name,
            abbr=_name,
            reader_cfg=bbeh_reader_cfg,
            infer_cfg=bbeh_infer_cfg.copy(),
            eval_cfg=bbeh_eval_cfg.copy()))