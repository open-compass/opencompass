from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import HeadQADataset


_hint = "The following questions come from exams to access a specialized position in the Spanish healthcare system. \n" \
    "Please choose the correct answer according to the question. \n"

HeadQA_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template="This is a {category} question which was extracted from the {year} {name} exam.\n" \
                "{qtext}\n{choices}Answer: {ra}",
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template={
            answer:
            f"{_hint}</E>This is a {{category}} question which was extracted from the {{year}} {{name}} exam.\n" \
            f"{{qtext}}\n{{choices}}Answer: {answer}"
            for answer in [1, 2, 3, 4, 5]
        },
        ice_token='</E>',
    ),
    retriever=dict(type=FixKRetriever, fix_id_list=[200, 400, 600, 800, 1000]),
    inferencer=dict(type=PPLInferencer))

HeadQA_eval_cfg = dict(evaluator=dict(type=AccEvaluator), )

langs = ['en', 'es']
HeadQA_datasets = []
for lang in langs:
    for _split in ['validation', 'test']:

        HeadQA_reader_cfg = dict(
            input_columns=['name', 'year', 'category', 'qtext', 'choices'],
            output_column='ra',
            test_split=_split
        )

        HeadQA_datasets.append(
            dict(
                abbr=f'HeadQA-{_split}',
                type=HeadQADataset,
                path='head_qa',
                name=lang,
                reader_cfg=HeadQA_reader_cfg,
                infer_cfg=HeadQA_infer_cfg,
                eval_cfg=HeadQA_eval_cfg
            )
        )
