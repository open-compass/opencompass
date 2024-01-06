from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import LogiQADataset


_hint = "The following are logical reasoning questions from the Chinese National " \
    "Civil Service Examination. Please choose the correct answer.\n"
LogiQA_en_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template="Context: {context}\nQuery: {query}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: {correct_option}",
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template={
            answer:
            f"{_hint}</E>Context: {{context}}\nQuery: {{query}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nAnswer: {answer}"
            for answer in ['A', 'B', 'C', 'D']
        },
        ice_token='</E>',
    ),
    retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
    inferencer=dict(type=PPLInferencer))

LogiQA_en_eval_cfg = dict(evaluator=dict(type=AccEvaluator), )


LogiQA_en_datasets = []
for _split in ["validation", "test"]:

    LogiQA_en_reader_cfg = dict(
        input_columns=['context', 'query', 'A', 'B', 'C', 'D'],
        output_column='correct_option',
        test_split=_split
    )

    LogiQA_en_datasets.append(
        dict(
            abbr=f'LogiQA_en-{_split}',
            type=LogiQADataset,
            path='lucasmccabe/logiqa',
            reader_cfg=LogiQA_en_reader_cfg,
            infer_cfg=LogiQA_en_infer_cfg,
            eval_cfg=LogiQA_en_eval_cfg
        )
    )
