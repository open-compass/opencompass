from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.arc_prize_public_evaluation import ARCPrizeDataset, ARCPrizeEvaluator


# The system_prompt defines the initial instructions for the model, 
# setting the context for solving ARC tasks.
system_prompt = '''You are a puzzle solving wizard. You are given a puzzle from the abstraction and reasoning corpus developed by Francois Chollet.'''

# User message template is a template for creating user prompts. It includes placeholders for training data and test input data, 
# guiding the model to learn the rule and apply it to solve the given puzzle.
user_message_template = '''Here are the example input and output pairs from which you should learn the underlying rule to later predict the output for the given test input:
----------------------------------------
{training_data}
----------------------------------------
Now, solve the following puzzle based on its input grid by applying the rules you have learned from the training data.:
----------------------------------------
[{{'input': {input_test_data}, 'output': [[]]}}]
----------------------------------------
What is the output grid? Only provide the output grid in the form as in the example input and output pairs. Do not provide any additional information:'''


arc_prize_public_evaluation_reader_cfg = dict(
    input_columns=['training_data', 'input_test_data'], 
    output_column='output_test_data'
)

arc_prize_public_evaluation_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='SYSTEM',fallback_role='HUMAN', prompt=system_prompt),
                dict(role='HUMAN', prompt=user_message_template),
            ],
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

arc_prize_public_evaluation_eval_cfg = dict(
    evaluator=dict(type=ARCPrizeEvaluator)
)

arc_prize_public_evaluation_datasets = [
    dict(
        abbr='ARC_Prize_Public_Evaluation',
        type=ARCPrizeDataset,
        path='opencompass/arc_prize_public_evaluation',
        reader_cfg=arc_prize_public_evaluation_reader_cfg,
        infer_cfg=arc_prize_public_evaluation_infer_cfg,
        eval_cfg=arc_prize_public_evaluation_eval_cfg
    )
]