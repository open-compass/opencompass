from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.utils.text_postprocessors import first_capital_postprocess
from opencompass.datasets import PromptCBLUEDataset

# 1. 子数据集名称
PromptCBLUE_lifescience_sets = [
    'CHIP-CDN', 'CHIP-CTC', 'KUAKE-QIC', 'IMCS-V2-DAC',
    'CHIP-STS', 'KUAKE-QQR', 'KUAKE-IR', 'KUAKE-QTR'
]

# 2. Reader 配置
reader_cfg = dict(
    input_columns=['input', 'answer_choices', 'options_str'],
    output_column='target',
    train_split='dev',
)

# 3. Prompt 模板：末行固定 ANSWER: $LETTER
_HINT = 'Given the ICD-10 candidate terms below, choose the normalized term(s) matching the original diagnosis.'

query_template = f"""{_HINT}

Original diagnosis: {{input}}

Options:
{{options_str}}

The last line of your response must be exactly:
ANSWER: $LETTER
""".strip()

infer_cfg_common = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[dict(role='HUMAN', prompt=query_template)]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

# 4. 评估配置：与 MMLU 同款
eval_cfg_common = dict(
    evaluator=dict(type=AccEvaluator),
    pred_postprocessor=dict(type=first_capital_postprocess),
)

# 5. 组装数据集配置
promptcblue_datasets = []
for ds_name in PromptCBLUE_lifescience_sets:
    promptcblue_datasets.append(dict(
        abbr=f'promptcblue_{ds_name.lower().replace("-", "_")}_norm',
        type=PromptCBLUEDataset,
        path='/fs-computility/ai4sData/shared/lifescience/tangcheng/LifeScience/opencompass_val/datasets/PromptCBLUE',
        name=ds_name,
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg_common,
        eval_cfg=eval_cfg_common,
    ))

# ★ OpenCompass 识别的出口变量
datasets = promptcblue_datasets
