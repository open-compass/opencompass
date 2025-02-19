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

base_prompt_en = """
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


base_prompt_cn = """
我希望你创建一个排行榜，用于评估来自各种大型语言模型的回答格式的正确性。为了完成这个任务，你将需要分析给模型的文本提示以及它们对应的回答。具体来说，请确保你的评估输出正确地格式化为JSON字符串。我将为此提供提示和回答。
以下是提示内容：
{
    "instruction": "{question}",
}
以下是模型的输出结果：
[
    {
        "model": "model",
        "answer": "{prediction}"
    },
]
请通过检查模型回答是否符合提示中声明的格式规范来评估模型回答的格式。进行彻底的格式检查，并提供格式正确或错误的详细解释。你的反馈应包括模型的名称，接着是格式正确性的状态，用'1'表示正确，'0'表示错误。将你的推理以每个评估模型的单个字符串中的 bullet 点形式呈现。换句话说，你应该生成以下输出：
```json
[
    {
        'model': <模型名称>,
        'format_correctness': <正确性>,
        'reasons': <格式正确性的原因>
    }
]
```
请注意，你的回答应是一个正确格式化的JSON字符串，不应包含任何额外的内容。我们将在Python中直接将其作为JSON字符串加载。
"""


fofo_datasets = []

for _name in subjective_all_sets:
    if '_cn' in _name:
        base_prompt = base_prompt_cn
    else:
        base_prompt = base_prompt_en
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
            inferencer=dict(type=GenInferencer),
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
