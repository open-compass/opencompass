from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.datasets import FofoDataset
from opencompass.summarizers import FofoSummarizer
from mmengine.config import read_base

subjective_reader_cfg = dict(
    input_columns=['question'],
    output_column='judge',
    )

subjective_all_sets = [
    'fofo_test_prompts', 'fofo_test_prompts_cn',
]

base_prompt = """
I would like you to create a leaderboard that evaluates the correctness of the format of answers from various large language models. To accomplish this, you will need to analyze the text prompts given to the models and their corresponding answers. Specifically, please ensure that your evaluation outputs are properly formatted as a json string. I will provide both the prompts and the responses for this purpose.

Here is the prompt:
{
    "instruction": "{question}",
}

Here are the outputs of the models:
[
    {
        "model": "model",
        "answer": "{prediction}"
    },
]

Please evaluate the formatting of the model's responses by checking if they comply with the format specifications stated in the prompt. Perform a thorough format check and provide a detailed explanation for why the format is correct or incorrect. Your feedback should include the name of the model, followed by the format correctness status represented as '1' for correct and '0' for incorrect. Present your reasoning as bullet points within a single string for each model assessed. In other words, you should produce the following output:
```json
[
    {
        'model': <model-name>,
        'format_correctness': <correctness>,
        'reasons': <reasons-of-format-correctness>
    }
]
```

Please note that your response should be a properly formatted JSON string and should not contain any additional content. We will load it directly as a JSON string in Python.
"""

fofo_datasets = []

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
                        prompt="You are a helpful assistant who evaluates the correctness and quality of models' outputs.")
                ],
                    round=[
                    dict(
                        role='HUMAN',
                        prompt = base_prompt
                    ),
                ]),
            ),
        ),
        pred_role='BOT',
    )

    fofo_datasets.append(
        dict(
            abbr=f'{_name}',
            type=FofoDataset,
            path='./data/subjective/fofo',
            name=_name,
            reader_cfg=subjective_reader_cfg,
            infer_cfg=subjective_infer_cfg,
            eval_cfg=subjective_eval_cfg,
            mode='singlescore',
            summarizer = dict(type=FofoSummarizer, judge_type='general')
        ))
