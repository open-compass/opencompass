from opencompass.datasets import BFCLASTEvaluator, BFCLDataset
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

# ---------------------------------------------------------------------------
# BFCL (Berkeley Function Calling Leaderboard) - single-turn AST evaluation.
# Dataset: gorilla-llm/Berkeley-Function-Calling-Leaderboard (v3)
# Paper  : https://arxiv.org/abs/2402.04653
# The dataset reader packs the function documentation into the system prompt
# (following the official BFCL prompt) and stores the ground truth needed by
# the AST evaluator in the ``gold`` column.
# ---------------------------------------------------------------------------

bfcl_reader_cfg = dict(
    input_columns=['system_prompt', 'user_prompt'],
    output_column='gold',
)

bfcl_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            dict(role='system', content='{system_prompt}'),
            dict(role='user', content='{user_prompt}'),
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

bfcl_eval_cfg = dict(
    evaluator=dict(type=BFCLASTEvaluator),
    pred_role='BOT',
)

bfcl_datasets = [
    dict(
        abbr='bfcl_simple',
        type=BFCLDataset,
        path='gorilla-llm/Berkeley-Function-Calling-Leaderboard',
        category='simple',
        reader_cfg=bfcl_reader_cfg,
        infer_cfg=bfcl_infer_cfg,
        eval_cfg=bfcl_eval_cfg,
    ),
    dict(
        abbr='bfcl_multiple',
        type=BFCLDataset,
        path='gorilla-llm/Berkeley-Function-Calling-Leaderboard',
        category='multiple',
        reader_cfg=bfcl_reader_cfg,
        infer_cfg=bfcl_infer_cfg,
        eval_cfg=bfcl_eval_cfg,
    ),
    dict(
        abbr='bfcl_parallel',
        type=BFCLDataset,
        path='gorilla-llm/Berkeley-Function-Calling-Leaderboard',
        category='parallel',
        reader_cfg=bfcl_reader_cfg,
        infer_cfg=bfcl_infer_cfg,
        eval_cfg=bfcl_eval_cfg,
    ),
    dict(
        abbr='bfcl_parallel_multiple',
        type=BFCLDataset,
        path='gorilla-llm/Berkeley-Function-Calling-Leaderboard',
        category='parallel_multiple',
        reader_cfg=bfcl_reader_cfg,
        infer_cfg=bfcl_infer_cfg,
        eval_cfg=bfcl_eval_cfg,
    ),
    dict(
        abbr='bfcl_live_simple',
        type=BFCLDataset,
        path='gorilla-llm/Berkeley-Function-Calling-Leaderboard',
        category='live_simple',
        reader_cfg=bfcl_reader_cfg,
        infer_cfg=bfcl_infer_cfg,
        eval_cfg=bfcl_eval_cfg,
    ),
    dict(
        abbr='bfcl_live_multiple',
        type=BFCLDataset,
        path='gorilla-llm/Berkeley-Function-Calling-Leaderboard',
        category='live_multiple',
        reader_cfg=bfcl_reader_cfg,
        infer_cfg=bfcl_infer_cfg,
        eval_cfg=bfcl_eval_cfg,
    ),
    dict(
        abbr='bfcl_live_parallel',
        type=BFCLDataset,
        path='gorilla-llm/Berkeley-Function-Calling-Leaderboard',
        category='live_parallel',
        reader_cfg=bfcl_reader_cfg,
        infer_cfg=bfcl_infer_cfg,
        eval_cfg=bfcl_eval_cfg,
    ),
    dict(
        abbr='bfcl_live_parallel_multiple',
        type=BFCLDataset,
        path='gorilla-llm/Berkeley-Function-Calling-Leaderboard',
        category='live_parallel_multiple',
        reader_cfg=bfcl_reader_cfg,
        infer_cfg=bfcl_infer_cfg,
        eval_cfg=bfcl_eval_cfg,
    ),
]
