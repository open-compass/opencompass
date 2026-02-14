from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import SWEBenchDataset

SWEBENCH_PROMPT = """You will be provided with a partial code base and an issue statement explaining a problem to resolve.

<issue>
{problem_statement}
</issue>

{hints_text}

Write a patch that resolves the issue. The patch should be in unified diff format.

<patch>
"""

swebench_reader_cfg = dict(
    input_columns=['repo', 'base_commit', 'problem_statement', 'hints_text'],
    output_column='patch',
)

swebench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt=SWEBENCH_PROMPT),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=4096),
)

swebench_eval_cfg = dict(
    evaluator=dict(
        type='SWEBenchEvaluator',
        dataset_name='princeton-nlp/SWE-bench',
        split='test',
        max_workers=4,
        timeout=1800,
        run_id='opencompass_swebench_eval',
        use_modal=False,
    ),
)

swebench_datasets = [
    dict(
        abbr='swebench',
        type=SWEBenchDataset,
        path='princeton-nlp/SWE-bench',
        split='test',
        max_problem_statement_length=10000,
        reader_cfg=swebench_reader_cfg,
        infer_cfg=swebench_infer_cfg,
        eval_cfg=swebench_eval_cfg,
    )
]
