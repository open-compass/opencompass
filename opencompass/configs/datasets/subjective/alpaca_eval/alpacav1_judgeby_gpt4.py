from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.datasets import SubjectiveCmpDataset
from mmengine.config import read_base

subjective_reader_cfg = dict(
    input_columns=['question'],
    output_column='judge',
    )

subjective_all_sets = [
    'alpaca_eval',
]


alpacav1_datasets = []

gpt4_prompt = """
I want you to create a leaderboard of different of large-language models. To do so, I will give you the instructions (prompts) given to the models, and the responses of two models. Please rank the models based on which responses would be preferred by humans. All inputs and outputs should be python dictionaries.

Here is the prompt:
{
    "instruction": "{question}"
}

Here are the outputs of the models:
[
    {
        "model": "model_1",
        "answer": "{prediction}"
    },
    {
        "model": "model_2",
        "answer": "{prediction2}"
    }
]

Now please rank the models by the quality of their answers, so that the model with rank 1 has the best output. Then return a list of the model names and ranks, i.e., produce the following output:
[
    {"model": <model-name>, "rank": <model-rank>},
    {"model": <model-name>, "rank": <model-rank>}
]

Your response must be a valid Python dictionary and should contain nothing else because we will directly execute it in Python. Please provide the ranking that the majority of humans would give.
"""


for _name in subjective_all_sets:
    subjective_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role='HUMAN',
                        prompt='{question}'
                    ),
                ]),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer, max_out_len=4096),
        )

    subjective_eval_cfg = dict(
        evaluator=dict(
            type=LMEvaluator,
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                begin=[
                    dict(
                        role='SYSTEM',
                        fallback_role='HUMAN',
                        prompt='You are a helpful assistant, that ranks models by the quality of their answers.')
                ],
                    round=[
                    dict(
                        role='HUMAN',
                        prompt = gpt4_prompt
                    ),
                ]),
            ),
        ),
        pred_role='BOT',
    )

    alpacav1_datasets.append(
        dict(
            abbr=f'{_name}',
            type=SubjectiveCmpDataset,
            path='./data/subjective/alpaca_eval',
            name=_name,
            reader_cfg=subjective_reader_cfg,
            infer_cfg=subjective_infer_cfg,
            eval_cfg=subjective_eval_cfg
        ))
