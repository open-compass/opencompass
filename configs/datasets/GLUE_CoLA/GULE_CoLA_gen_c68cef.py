from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import CoLADataset
from opencompass.utils.text_postprocessors import first_option_postprocess


CoLA_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role="HUMAN",
                    prompt="Sentence: {sentence}\nQuestion: Whether the above sentence is linguistically acceptable?\nA. unacceptable\nB. acceptable\nAnswer: "
                ),
            ], ),
        ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

CoLA_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator), 
    pred_role="BOT",
    pred_postprocessor=dict(type=first_option_postprocess, options='AB'),)

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





