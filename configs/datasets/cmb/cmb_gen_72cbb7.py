from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import CMBDataset


cmb_datasets = []

cmb_reader_cfg = dict(
    input_columns=["exam_type", "exam_class", "question_type", "question", "option_str"],
    output_column=None,
    train_split="val",
    test_split="test"
)

cmb_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(
            begin="</E>",
            round=[
                dict(
                    role="HUMAN",
                    prompt=f"以下是中国{{exam_type}}中{{exam_class}}考试的一道{{question_type}}，不需要做任何分析和解释，直接输出答案选项。\n{{question}}\n{{option_str}} \n 答案: ",
                ),
                dict(role="BOT", prompt="{answer}"),
            ],
        ),
        ice_token="</E>",
    ),
    retriever=dict(type=FixKRetriever),
    inferencer=dict(type=GenInferencer, fix_id_list=[0, 1, 2, 3, 4]),
)

cmb_datasets.append(
    dict(
        type=CMBDataset,
        path="./data/CMB/",
        abbr="cmb",
        reader_cfg=cmb_reader_cfg,
        infer_cfg=cmb_infer_cfg
    )
)