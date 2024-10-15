from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GaoKaoMATHDataset, GaoKaoMATHEvaluator


MATH_CN_PROMPT="""
你是一个数学阅卷专家，任务是从给定的回答句子中提取精确的关键答案。你必须只提供提取的关键答案，不包括任何额外的文字。
—
我将为你提供一个问题、回答句子和问题类型。回答句子是对所提供问题的回应。利用提供的信息，你必须准确而精确地确定并从回答句子中提取预期的关键答案。请不要对问题发表主观看法。

对于单选题，答案应该是选项字母，例如 "A"；
对于多选题，答案应该是一个选项字母的列表，例如 ["A"] 或 ["A", "B", "C"]；
对于填空题，答案应该是一个填入空白处的答案列表，列表的数量应该与问题中的空白数量相同，例如 ["$$\\frac{{1}}{{2}}$$"] 或 ["$$\\frac{{1}}{{2}}$$", "2"]。
对于问答题，类似填空题，为每个小问抽出相应答案，例如 ["$$\\frac{{1}}{{2}}$$"] 或 ["$$\\frac{{1}}{{2}}$$", "2"]。

如果回答句子提供了多个不同的答案，请仔细判断后面提供的答案是否是对前面答案的修正或修改。如果是这样，提取这个修正或修改后的答案作为最终答案。相反，如果回答句子在多个答案之间波动而没有明确的最终答案，你应该输出 [No valid answer]。
—
问题类型: {question_type}
原始问题: {question}
回答: {response}
提取的关键答案:
"""

gaokao_math_reader_cfg = dict(input_columns=['question', 'response', 'question_type'], output_column='extract_answer')


gaokao_math_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(role='HUMAN', prompt=MATH_CN_PROMPT),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512))

gaokao_math_eval_cfg = dict(
    evaluator=dict(type=GaoKaoMATHEvaluator, model_name='Qwen/Qwen2.5-72B-Instruct', url=['http://22.8.73.119:23333/v1', 'http://22.8.4.97:23333/v1', 'http://22.8.22.254:23333/v1', 'http://22.8.17.14:23333/v1']))

gaokao_math_datasets = [
    dict(
        type=GaoKaoMATHDataset,
        abbr='GaoKaoMATH',
        path='./data/gaokao_math/test_2k.json',
        reader_cfg=gaokao_math_reader_cfg,
        infer_cfg=gaokao_math_infer_cfg,
        eval_cfg=gaokao_math_eval_cfg)
]
