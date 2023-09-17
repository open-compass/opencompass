from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import CoLADataset


CoLA_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            0: 'Please determine whether the following sentence is linguistically acceptable.\nSentence: {sentence}\nResult: unacceptable.',
            1: 'Please determine whether the following sentence is linguistically acceptable.\nSentence: {sentence}\nResult: acceptable.'
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

CoLA_eval_cfg = dict(evaluator=dict(type=AccEvaluator), )

CoLA_datasets = []
for _split in ["in_domain", "out_of_domain"]:

    CoLA_reader_cfg = dict(
        input_columns=['sentence'],
        output_column='label',
        test_split='dev'
    )

    CoLA_datasets.append(
        dict(
            abbr=f'CoLA-{_split}',
            type=CoLADataset,
            path=f'./data/GLUE/CoLA',
            name=_split,
            reader_cfg=CoLA_reader_cfg,
            infer_cfg=CoLA_infer_cfg,
            eval_cfg=CoLA_eval_cfg
        )
    )
        