from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.datasets import (PerspectiveGapDataset,
                                  PerspectiveGapRoleAssignmentEvaluator)

perspectivegap_role_assignment_reader_cfg = dict(
    input_columns=['input'],
    output_column='reference_data',
)

perspectivegap_role_assignment_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[dict(role='HUMAN', prompt='{input}')],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=8192),
)

perspectivegap_role_assignment_eval_cfg = dict(
    evaluator=dict(type=PerspectiveGapRoleAssignmentEvaluator),
    pred_role='BOT',
)

perspectivegap_role_assignment_datasets = [
    dict(
        type=PerspectiveGapDataset,
        path='sun1245/PerspectiveGap',
        name='role_assignment',
        abbr='perspectivegap-role_assignment',
        reader_cfg=perspectivegap_role_assignment_reader_cfg,
        infer_cfg=perspectivegap_role_assignment_infer_cfg,
        eval_cfg=perspectivegap_role_assignment_eval_cfg,
    )
]
