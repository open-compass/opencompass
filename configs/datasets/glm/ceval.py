from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import GLMChoiceInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import CEvalDataset

ceval_reader_cfg = dict(
    input_columns=['question', 'A', 'B', 'C', 'D'],
    output_column='answer',
    train_split='dev',
    test_split="val")

ceval_prompt_template = dict(
    type=PromptTemplate,
    template=None,
    ice_token='</E>',
)

ceval_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template={
            answer:
            f'{{question}}\n(A) {{/A}}\n(B) {{/B}}\n(C) {{/C}}\n(D) {{/D}}\n答案: ({answer}) {{{answer}}}\n'
            for answer in ['A', 'B', 'C', 'D']
        }),
    prompt_template=ceval_prompt_template,
    retriever=dict(type=FixKRetriever),
    inferencer=dict(type=GLMChoiceInferencer, fix_id_list=[0, 1, 2, 3, 4]))

ceval_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

ceval_all_sets = [
    "操作系统",
    "初中地理",
    "初中化学",
    "初中历史",
    "初中生物",
    "初中数学",
    "初中物理",
    "初中政治",
    "大学编程",
    "大学化学",
    "大学经济学",
    "大学物理",
    "大学中国史",
    "导游资格",
    "法律职业资格",
    "法学",
    "概率统计",
    "高等数学",
    "高中地理",
    "高中化学",
    "高中历史",
    "高中生物",
    "高中数学",
    "高中物理",
    "高中语文",
    "高中政治",
    "公务员",
    "工商管理",
    "环境影响评价工程师",
    "基础医学",
    "计算机网络",
    "计算机组成",
    "教师资格",
    "教育学",
    "离散数学",
    "临床医学",
    "逻辑学",
    "马克思主义基本原理",
    "毛泽东思想和中国特色社会主义理论体系概论",
    "兽医学",
    "税务师",
    "思想道德修养与法律基础",
    "体育学",
    "医师资格",
    "艺术学",
    "植物保护",
    "中国语言文学",
    "注册城乡规划师",
    "注册电气工程师",
    "注册会计师",
    "注册计量师",
    "注册消防工程师",
]

ceval_datasets = []
for _name in ceval_all_sets:
    ceval_datasets.append(
        dict(
            type=CEvalDataset,
            path="./data/ceval/release_ceval",
            name=_name,
            abbr='ceval-' + _name,
            reader_cfg=ceval_reader_cfg,
            infer_cfg=ceval_infer_cfg.copy(),
            eval_cfg=ceval_eval_cfg.copy()))

    ceval_datasets[-1]['infer_cfg'][
        'prompt_template'] = ceval_prompt_template.copy()
    ceval_datasets[-1]['infer_cfg']['prompt_template']['template'] = dict(
        begin=[
            dict(
                role='SYSTEM',
                fallback_role='HUMAN',
                prompt=f'以下是中国关于{_name}考试的单项选择题，请选出其中的正确答案。'),
            '</E>',
        ],
        round=[
            dict(
                role='HUMAN',
                prompt=
                '{question}\n(A) {A}\n(B) {B}\n(C) {C}\n(D) {D}\答案: ('),
        ],
    )

del _name
