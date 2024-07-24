from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import JiebaRougeEvaluator
from opencompass.datasets.rolebench import InstructionGeneralizationChineseDataset

instruction_generalization_zh_reader_cfg = dict(
    input_columns=['role', 'desc', 'question'],
    output_column='answer',
    train_split='train',
    test_split='test'
)

instruction_generalization_zh_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(role='SYSTEM', fallback_role='HUMAN', prompt='你是{role}，你的特征描述是：{desc}。现在请你回答我的一些问题，以准确展现你的人格特征！你的说话风格要全面模仿被赋予的人格角色！请不要暴露你是人工智能模型或者语言模型，你要时刻记住你只被赋予的一个人格角色。说话不要嗦，也不要太过于正式或礼貌。'),
            ],
            round=[
                dict(role='HUMAN', prompt='{question}'),
                dict(role='BOT', prompt=''),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512)
)

instruction_generalization_zh_eval_cfg = dict(
    evaluator=dict(type=JiebaRougeEvaluator),
    pred_role='BOT'
)

instruction_generalization_zh_datasets = [
    dict(
        type=InstructionGeneralizationChineseDataset,
        path='ZenMoore/RoleBench',
        reader_cfg=instruction_generalization_zh_reader_cfg,
        infer_cfg=instruction_generalization_zh_infer_cfg,
        eval_cfg=instruction_generalization_zh_eval_cfg)
]
