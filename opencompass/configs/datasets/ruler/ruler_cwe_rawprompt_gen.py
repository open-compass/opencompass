from opencompass.datasets.ruler.ruler_cwe import RulerCweDataset
from opencompass.datasets.ruler.ruler_cwe import RulerCweEvaluator
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer

# CWE Dataset
cwe_datasets = [
    {
        'abbr': 'ruler_cwe',
        'type': RulerCweDataset,
        'freq_cw': 30,
        'freq_ucw': 3,
        'num_cw': 10,
        'tokens_to_generate': 120,
        'reader_cfg': dict(input_columns=['prompt'], output_column='answer'),
        'infer_cfg': dict(
            prompt_template=dict(
                type=RawPromptTemplate,
                messages=[
                    {'role': 'user', 'content': '{prompt}'},
                ],
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer),
        ),
        'eval_cfg': dict(
            evaluator=dict(type=RulerCweEvaluator),
        ),
    }
]
