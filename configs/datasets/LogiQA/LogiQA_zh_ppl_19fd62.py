from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import LogiQADataset


_hint = "以下是中国国家公务员考试的逻辑推理题，请选出其中的正确答案。\n"
LogiQA_zh_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template="上下文信息: {context}\n提问: {query}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n答案: {correct_option}",
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template={
            answer:
            f"{_hint}</E>上下文信息: {{context}}\n提问: {{query}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\n答案: {answer}"
            for answer in ['A', 'B', 'C', 'D']
        },
        ice_token='</E>',
    ),
    retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
    inferencer=dict(type=PPLInferencer))

LogiQA_zh_eval_cfg = dict(evaluator=dict(type=AccEvaluator), )


LogiQA_zh_datasets = []
for _split in ["validation", "test"]:

    LogiQA_zh_reader_cfg = dict(
        input_columns=['context', 'query', 'A', 'B', 'C', 'D'],
        output_column='correct_option',
        test_split=_split
    )

    LogiQA_zh_datasets.append(
        dict(
            abbr=f'LogiQA_zh-{_split}',
            type=LogiQADataset,
            path='jiacheng-ye/logiqa-zh',
            reader_cfg=LogiQA_zh_reader_cfg,
            infer_cfg=LogiQA_zh_infer_cfg,
            eval_cfg=LogiQA_zh_eval_cfg
        )
    )
