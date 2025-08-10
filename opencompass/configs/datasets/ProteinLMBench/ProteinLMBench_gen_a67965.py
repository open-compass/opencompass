from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets.ProteinLMBench import ProteinLMBenchDataset, ProteinLMBenchEvaluator

QUERY_TEMPLATE = "Answer the following multiple choice question. There is only one correct answer. The last line of your response should be in the format 'Answer: $LETTER' (without quotes), where LETTER is the letter among {start} through {end}.\n{question}"


# Reader configuration
reader_cfg = dict(
    input_columns=['question', 'start', 'end', 'options'],
    output_column='label',
)

# Inference configuration
infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt=QUERY_TEMPLATE
                )
            ], ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

# Evaluation configuration
eval_cfg = dict(
    evaluator=dict(type=ProteinLMBenchEvaluator),
)

proteinlmbench_dataset = dict(
    abbr='ProteinLMBench',
    type=ProteinLMBenchDataset,
    path='tsynbio/ProteinLMBench',
    reader_cfg=reader_cfg,
    infer_cfg=infer_cfg,
    eval_cfg=eval_cfg
)

proteinlmbench_datasets = [proteinlmbench_dataset]
