from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import ZebraLogicDataset
from opencompass.datasets.zebralogic import ZebraLogicMCEvaluator

# ---------------------------------------------------------------------------
# ZebraLogic – mc_mode
# Multiple-choice questions derived from zebra logic puzzles.
# Each puzzle yields one question; the model must select the correct answer
# letter (A, B, C, …) from the formatted choices.
# Dataset: WildEval/ZebraLogic  (config: mc_mode, 3259 test samples)
# Paper  : https://arxiv.org/abs/2502.01100
# ---------------------------------------------------------------------------

zebralogic_mc_reader_cfg = dict(
    input_columns=['puzzle', 'question', 'formatted_choices'],
    output_column='answer_label',
)

_MC_PROMPT = (
    'The following is a logic grid puzzle. Read all the clues carefully and '
    'answer the question.\n\n'
    'Puzzle:\n{puzzle}\n\n'
    'Question: {question}\n\n'
    'Choices:\n{formatted_choices}\n\n'
    'Think step by step. When you provide the final answer, use the prefix '
    '"The answer is:" followed by only the answer letter (e.g., '
    '"The answer is: A"). Do not include any other text after the answer.'
)

zebralogic_mc_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            dict(role='user', content=_MC_PROMPT),
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

zebralogic_mc_eval_cfg = dict(
    evaluator=dict(type=ZebraLogicMCEvaluator),
    pred_role='BOT',
)

# ---------------------------------------------------------------------------
# ZebraLogic – grid_mode
# Full grid completion: the model must output the entire solution table.
# Evaluated with cell-level accuracy.
# Dataset: WildEval/ZebraLogic  (config: grid_mode, 1000 test samples)
# ---------------------------------------------------------------------------
from opencompass.datasets.zebralogic import ZebraLogicGridEvaluator  # noqa: E402

zebralogic_grid_reader_cfg = dict(
    input_columns=['puzzle'],
    output_column='solution',
)

_GRID_PROMPT = (
    'The following is a logic grid puzzle. Read all the clues carefully and '
    'fill in the complete solution table.\n\n'
    '{puzzle}\n\n'
    'Output the solution as a markdown table where the first column is '
    '"House" (numbered 1 to N) and the remaining columns correspond to the '
    'attributes. Every cell must be filled. Do not output anything after the '
    'table.'
)

zebralogic_grid_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            dict(role='user', content=_GRID_PROMPT),
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

zebralogic_grid_eval_cfg = dict(
    evaluator=dict(type=ZebraLogicGridEvaluator),
    pred_role='BOT',
)

# ---------------------------------------------------------------------------
# Combined dataset list
# ---------------------------------------------------------------------------

zebralogic_datasets = [
    dict(
        type=ZebraLogicDataset,
        abbr='zebralogic_mc',
        path='WildEval/ZebraLogic',
        config='mc_mode',
        reader_cfg=zebralogic_mc_reader_cfg,
        infer_cfg=zebralogic_mc_infer_cfg,
        eval_cfg=zebralogic_mc_eval_cfg,
    ),
    dict(
        type=ZebraLogicDataset,
        abbr='zebralogic_grid',
        path='WildEval/ZebraLogic',
        config='grid_mode',
        reader_cfg=zebralogic_grid_reader_cfg,
        infer_cfg=zebralogic_grid_infer_cfg,
        eval_cfg=zebralogic_grid_eval_cfg,
    ),
]
