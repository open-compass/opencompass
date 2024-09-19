from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.ruler.ruler_fwe import RulerFweDataset
from opencompass.datasets.ruler.ruler_fwe import RulerFweEvaluator

# FWE Dataset
fwe_datasets = [
    {
        'abbr': 'ruler_fwe',
        'type': RulerFweDataset,
        'tokens_to_generate': 50,
        'alpha': 2.0,
        'coded_wordlen': 6,
        'reader_cfg': dict(input_columns=['prompt'], output_column='answer'),
        'infer_cfg': dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    round=[
                        dict(role='HUMAN', prompt='{prompt}'),
                        dict(role='BOT', prompt='{answer}\n'),
                    ]
                ),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer),
        ),
        'eval_cfg': dict(
            evaluator=dict(type=RulerFweEvaluator),
        ),
    }
]
