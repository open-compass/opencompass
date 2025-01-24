from opencompass.datasets import WildBenchDataset
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.openicl.icl_inferencer import ChatInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

hu_life_qa_reader_cfg = dict(
    input_columns=["dialogue", "prompt"],
    output_column="judge",
)

data_path ="/mnt/hwfile/opendatalab/yanghaote/share/g13k_hu/g13k_hu_vpaper.jsonl"

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
    "business and finance": "business and finance",
    "childbearing and education": "life, culture, and customs",
    "culture and community": "life, culture, and customs",
    'culture and customs': "life, culture, and customs",
    "life, culture, and customs": "life, culture, and customs",
    "education and profession": "education and profession",
    "food and drink": "life, culture, and customs",
    "health": "life, culture, and customs",
    "holidays": "life, culture, and customs",
    "home": "life, culture, and customs",
    "person": "life, culture, and customs",
    "politics": "politics, policy and law",
    "politics, policy and law": "politics, policy and law",
    "public education and courses": "education and profession",
    "transport": "life, culture, and customs",
    "science": "life, culture, and customs",
    "travel": "life, culture, and customs",
}
