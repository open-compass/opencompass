from opencompass.datasets import MaritimeBenchDataset
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.utils.text_postprocessors import first_option_postprocess
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer

maritimebench_reader_cfg = dict(
    input_columns=['question', 'A', 'B', 'C', 'D'],
    output_column='answer',
    train_split='test'  # 明确指定使用test分割
)

maritimebench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='请回答单选题。要求只输出选项，不输出解释，将选项放在<>里，直接输出答案。示例：\n\n题目：在船舶主推进动力装置中，传动轴系在运转中承受以下复杂的应力和负荷，但不包括______。\n选项：\nA. 电磁力\nB. 压拉应力\nC. 弯曲应力\nD. 扭应力\n答：<A> 当前题目:\n {question}\nA:{A}\nB:{B}\nC:{C}\nD:{D}')
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),  # 不使用上下文
    inferencer=dict(type=GenInferencer)  # 添加推理器配置
)

maritimebench_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD')
)

maritimebench_datasets = [
    dict(
        abbr='maritimebench',
        type=MaritimeBenchDataset,
        name='default',
        path='opencompass/maritimebench',
        reader_cfg=maritimebench_reader_cfg,
        infer_cfg=maritimebench_infer_cfg,
        eval_cfg=maritimebench_eval_cfg
    )
]