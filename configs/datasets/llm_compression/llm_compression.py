from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import SWCELossInferencer
from opencompass.openicl.icl_evaluator import BPCEvaluator
from opencompass.datasets import LLMCompressionDataset


# The three corpora for llm_compression used in the original paper
# See configs/datasets/llm_compression/README.md for more details
subset_mapping = {
    'arxiv_math': ['arxiv_math'],
    'commoncraw': ['cc'],
    'python': ['python'],
}


# Build LLM Compression datasets
llm_compression_datasets = []
for _name in subset_mapping.keys():
    llm_cmp_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template='{content}',
        ),
        # No in-context example, using ZeroRetriever
        retriever=dict(type=ZeroRetriever),
        # Calculates cross entropy loss for each batch based on a sliding context window
        # Setting block_size=1900 and stride=512 according to the original paper
        inferencer=dict(type=SWCELossInferencer, block_size=1900, stride=512),
    )

    # Calculates Bits per Character (BPC) based on the CE loss from the inference stage
    llm_cmp_eval_cfg = dict(evaluator=dict(type=BPCEvaluator))

    llm_compression_datasets.append(
        dict(
            abbr=f'llm_compression-{_name}',
            type=LLMCompressionDataset,
            path='./data/llm-compression',
            name=_name,
            samples=None,  # Set small samples for testing
            reader_cfg=dict(
                input_columns=['content'],
                output_column=None,
            ),
            infer_cfg=llm_cmp_infer_cfg,
            eval_cfg=llm_cmp_eval_cfg,
        ))

del _name
