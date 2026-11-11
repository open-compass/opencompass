from opencompass.datasets import (WeddooBenchDataset,
                                  generic_llmjudge_postprocess)
from opencompass.evaluator import GenericLLMEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

QUERY_TEMPLATE = """
你是一位婚嫁行业的资深婚礼顾问。请回答下面这位客户的问题，给出具体、可执行的建议。

要求：
- 先给结论，再给理由和操作步骤；
- 涉及价格时不要编造具体金额，说明区间或以门店报价为准；
- 不确定的信息不要臆造，可以说明需要补充哪些信息。

客户问题：
{question}
""".strip()

GRADER_TEMPLATE = """
你是一位婚嫁行业的资深评审专家。请判断候选回答是否在语义上正确回答了客户问题。

评分依据：
1. 参考答案与核心要点均由婚嫁行业专家审定，视为正确。你不需要重新回答问题，只做判定。
2. 判定标准：候选回答覆盖的核心要点达到 60% 及以上，且不存在事实性错误，判为 A（正确）。
3. 覆盖不足 60%，或存在事实性错误（例如答错城市、编造价格、杜撰政策流程），判为 B（错误）。
4. 措辞、语气、结构、详略程度可以与参考答案不同。只要语义正确即判 A，不要因为回答没有逐字出现参考答案而判 B。
5. 回答空洞、答非所问、只做无信息量的寒暄或推诿，判 B。
6. 涉及价格时：给出区间、给出参考范围、或说明以门店报价为准，都算正确；编造具体金额且与参考答案冲突的判 B。

只需要输出一个大写字母：A 或 B。
不要输出任何其他文字、解释、标点或空格。

<客户问题>
{question}
</客户问题>

<参考答案>
{answer}
</参考答案>

<核心要点>
{key_points_text}
</核心要点>

<候选回答>
{prediction}
</候选回答>
""".strip()

weddoo_bench_reader_cfg = dict(
    input_columns=['question'],
    output_column='answer',
)

weddoo_bench_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            dict(role='user', content=QUERY_TEMPLATE),
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

weddoo_bench_eval_cfg = dict(
    evaluator=dict(
        type=GenericLLMEvaluator,
        prompt_template=dict(
            type=RawPromptTemplate,
            messages=[
                dict(
                    role='system',
                    content='你是一位严谨的评审专家，只按评分标准判定，不重新回答问题。',
                ),
                dict(role='user', content=GRADER_TEMPLATE),
            ],
        ),
        dataset_cfg=dict(
            type=WeddooBenchDataset,
            path='opencompass/WeddooBench',
            reader_cfg=weddoo_bench_reader_cfg,
        ),
        judge_cfg=dict(),
        dict_postprocessor=dict(type=generic_llmjudge_postprocess),
    ),
    pred_role='BOT',
)

weddoo_bench_datasets = [
    dict(
        abbr='WeddooBench',
        type=WeddooBenchDataset,
        path='opencompass/WeddooBench',
        reader_cfg=weddoo_bench_reader_cfg,
        infer_cfg=weddoo_bench_infer_cfg,
        eval_cfg=weddoo_bench_eval_cfg,
    )
]
