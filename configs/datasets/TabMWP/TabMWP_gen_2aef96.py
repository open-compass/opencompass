from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import TabMWPDataset

# None of the TabMWP dataset in huggingface is correctly parsed, so we use our own dataset reader
# Please download the dataset from https://github.com/lupantech/PromptPG/tree/main

input_format='TQ'
output_format='A'
elements = {"Q": f"Question: {{question}}", 
            "T": f"Table: {{table}}", 
            "S": f"Solution: {{solution}}",
            "A": f"Answer: The answer is {{answer}}.",
            "AS": f"Answer: The answer is {{answer}}. BECAUSE: {{solution}}",
            "SA": f"Answer: {{solution}} The answer is {{answer}}."}


TabMWP_reader_cfg = dict(
    input_columns=["question", "table"],
    output_column="answer", #choose from ["answer"，"solution"，"answer_and_solution"，"solution_and_answer"]
    train_split='dev',
    )

TabMWP_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role="HUMAN",
                    prompt= "\n".join(elements[label] for label in input_format)
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

TabMWP_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
)

TabMWP_datasets = [
    dict(
        type=TabMWPDataset,
        path="./data/tabmwp/",
        reader_cfg=TabMWP_reader_cfg,
        infer_cfg=TabMWP_infer_cfg,
        eval_cfg=TabMWP_eval_cfg,)
]

