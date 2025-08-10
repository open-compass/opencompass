from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import ChatInferencer
from opencompass.openicl.icl_evaluator import TEvalEvaluator
from opencompass.datasets import teval_postprocess, TEvalDataset

plugin_eval_subject_mapping = {
    'instruct': ['instruct_v1'],
    'instruct_zh': ['instruct_v1_zh'],
    'plan': ['plan_json_v1', 'plan_str_v1'],
    'plan_zh': ['plan_json_v1_zh', 'plan_str_v1_zh'],
    'review': ['review_str_v1'],
    'review_zh': ['review_str_v1_zh'],
    'reason_retrieve_understand': ['reason_retrieve_understand_json_v1'],
    'reason_retrieve_understand_zh': ['reason_retrieve_understand_json_v1_zh'],
    'reason': ['reason_str_v1'],
    'reason_zh': ['reason_str_v1_zh'],
    'retrieve': ['retrieve_str_v1'],
    'retrieve_zh': ['retrieve_str_v1_zh'],
    'understand': ['understand_str_v1'],
    'understand_zh': ['understand_str_v1_zh'],
}

plugin_eval_datasets = []
for _name in plugin_eval_subject_mapping:
    plugin_eval_reader_cfg = dict(input_columns=['prompt'], output_column='ground_truth')
    plugin_eval_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(role='HUMAN', prompt='{prompt}'),
                ],
            ),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=ChatInferencer),
    )
    plugin_eval_eval_cfg = dict(
        evaluator=dict(type=TEvalEvaluator, subset=_name),
        pred_postprocessor=dict(type=teval_postprocess),
        num_gpus=1,
    )

    for subset in plugin_eval_subject_mapping[_name]:
        plugin_eval_datasets.append(
            dict(
                abbr='plugin_eval-mus-p10-' + subset + '_public',
                type=TEvalDataset,
                path='data/compassbench_v1.1.public/agent-teval-p10',
                name=subset,
                reader_cfg=plugin_eval_reader_cfg,
                infer_cfg=plugin_eval_infer_cfg,
                eval_cfg=plugin_eval_eval_cfg,
            )
        )
