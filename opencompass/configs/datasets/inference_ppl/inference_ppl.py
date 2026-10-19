from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import InferencePPLOnlyInferencer
from opencompass.openicl.icl_evaluator import AverageInferencePPLEvaluator

from opencompass.datasets import InferencePPLDataset

# Build InferencePPLDataset
inference_ppl_datasets = []

llm_cmp_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='{text}',
    ),
    # No in-context example, using ZeroRetriever
    retriever=dict(type=ZeroRetriever),
    # compute inference-ppl
    inferencer=dict(type=InferencePPLOnlyInferencer),
)

# Average the inference-ppl scores
llm_cmp_eval_cfg = dict(evaluator=dict(type=AverageInferencePPLEvaluator))

inference_ppl_datasets.append(
    dict(
        abbr=f'inference-ppl',
        type=InferencePPLDataset,
        path='./data/inference_ppl',
        name='cn-reasoning-val',
        samples=None,  # Set small samples for testing
        reader_cfg=dict(
            input_columns=['text'],
            output_column=None,
        ),
        infer_cfg=llm_cmp_infer_cfg,
        eval_cfg=llm_cmp_eval_cfg,
    ))
