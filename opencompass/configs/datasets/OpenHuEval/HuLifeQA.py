from opencompass.datasets import WildBenchDataset
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.openicl.icl_inferencer import ChatInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

hu_life_qa_reader_cfg = dict(
    input_columns=["dialogue", "prompt"],
    output_column="judge",
)

data_path ="/mnt/hwfile/opendatalab/yanghaote/share/HuLifeQA_20250131.jsonl"

hu_life_qa_datasets = []
hu_life_qa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="""{dialogue}"""
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(
        type=ChatInferencer,
        max_seq_len=4096,
        max_out_len=512,
        infer_mode="last",
    ),
)

hu_life_qa_eval_cfg = dict(
    evaluator=dict(
        type=LMEvaluator,
        prompt_template=dict(
            type=PromptTemplate, 
            template="""{prompt}"""
        ),
    ),
    pred_role="BOT",
)

hu_life_qa_datasets.append(
    dict(
        abbr="hu_life_qa",
        type=WildBenchDataset,
        path=data_path,
        reader_cfg=hu_life_qa_reader_cfg,
        infer_cfg=hu_life_qa_infer_cfg,
        eval_cfg=hu_life_qa_eval_cfg,
    )
)

task_group_new = {
    "life_culture_custom": "life_culture_custom",
    "childbearing and education": "life_culture_custom",
    "culture and community": "life_culture_custom",
    'culture and customs': "life_culture_custom",
    "food and drink": "life_culture_custom",
    "health": "life_culture_custom",
    "holidays": "life_culture_custom",
    "home": "life_culture_custom",
    "person": "life_culture_custom",
    "transport": "life_culture_custom",
    "science": "life_culture_custom",
    "travel": "life_culture_custom",
    "business_finance": "business_finance",
    "business and finance": "business_finance",
    "education_profession": "education_profession",
    "public education and courses": "education_profession",
    "politics_policy_law": "politics_policy_law",
    "politics": "politics_policy_law",
}
