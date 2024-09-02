from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.datasets import CompassArenaDataset

subjective_reader_cfg = dict(
    input_columns=['question', 'ref'],
    output_column='judge',
    )

data_path ='data/subjective/compass_arena'

subjective_datasets = []

base_prompt = """

[回答1开始]
{prediction}
[回答1结束]

[回答2开始]
{prediction2}
[回答2结束]

根据评分要求，在以下 3 个选项中做出选择:
A. 回答1更好
B. 回答2更好
C. 回答1、2平局
并提供你的解释原因。

如果你认为回答1更好，你的输出应形如：
选择：A
原因：blahblah blahblah\n

如果你认为回答2更好，你的输出应形如：
选择：B
原因：blahblah blahblah\n

如果你认为回答1、2打成平手，你的输出应形如：
选择：C
原因：blahblah blahblah\n
"""

knowledge_prompt = """
请根据提供的 评分要求，用户问题，参考答案 以及 相应的两个回答（回答1，回答2），判断两个回答中哪一个更好。
评分要求（重要性依次递减）:
1. 更好的回答能与参考答案吻合或表明参考答案的意思。
2. 在都准确答对问题的前提下，更好的回答能对知识点进行额外补充，且补充的知识准确无误。
3. 更好的回答更加符合与人类对话的习惯，包括语气、情调等。

[用户问题]
{question}

[参考答案]
{ref}
""" + base_prompt


language_prompt = """
请根据提供的 评分要求，用户问题 以及 相应的两个回答（回答1，回答2），判断两个回答中哪一个更好。
评分要求（重要性依次递减）:
1. 在有明确的参考答案的情况下，越贴近参考答案或表明了参考答案的意思的回答越好。
2. 更好的回答在语言表达上更流畅，更加符合与人类对话的习惯，包括语气、情调等
3. 在都准确答对问题的前提下，更好的回答能进行额外补充，且补充的内容准确无误。

[用户问题]
{question}

[参考答案]
{ref}
""" + base_prompt


math_prompt = """
请根据提供的 评分要求，用户问题，参考答案 以及 相应的两个回答（回答1，回答2），判断两个回答中哪一个更好。
评分要求（重要性依次递减）:
1. 更好的回答的答案能和参考答案一致。
2. 若两个回答的答案都与参考答案不一致，则更好的回答的推理过程应更加合理。
3. 更好的回答更加符合与人类对话的习惯，包括语气、情调等。

[用户问题]
{question}

[参考答案]
{ref}
""" + base_prompt

reason_prompt = math_prompt

creation_prompt = """
请根据提供的 评分要求，用户问题 以及 相应的两个回答（回答1，回答2），判断两个回答中哪一个更好。
评分要求（重要性依次递减）:
1. 好的回答必须首先符合用户问题里的各种需求，不能跑题
2. 好的回答必须具有逻辑连贯性，围绕一个中心进行回答
3. 好的回答必须具有创造性的词语和表达丰富度

[用户问题]
{question}
""" + base_prompt

sub_map = {'creationv3': creation_prompt}

for _name, _prompt in sub_map.items():
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
            inferencer=dict(type=GenInferencer, max_seq_len=4096, max_out_len=2048),
        )

    subjective_eval_cfg = dict(
        evaluator=dict(
            type=LMEvaluator,
            infer_order='double',
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role='HUMAN',
                        prompt = _prompt
                    ),
                ]),
            ),
        ),
        pred_role='BOT',
    )

    subjective_datasets.append(
        dict(
            abbr=f'{_name}',
            type=CompassArenaDataset,
            path=data_path,
            name=_name,
            reader_cfg=subjective_reader_cfg,
            infer_cfg=subjective_infer_cfg,
            eval_cfg=subjective_eval_cfg
        ))
