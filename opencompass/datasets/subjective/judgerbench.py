# flake8: noqa
import json
import os.path as osp
import re

import numpy as np
import pandas as pd
from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset

#######################Judgerbench A Prompt#############################

base_prompt_prefix_en = """Please read the dialogue between the two assistants and the user to determine which assistant performed better during the conversation.
Here is the dialogue content:
[Dialogue Begin]\n"""
### User:
### Assistant A:
### Assistant B:
base_prompt_suffix_en = """\n[Dialogue End]
If you believe Assistant A performed better, please output A directly.
If you believe Assistant B performed better, please output B directly.
Do not output any other content, just the option.
Please output:"""

base_prompt_prefix_cn = """请阅读两个助手与用户之间的对话，以确定在对话过程中哪个助手表现更好。
以下是对话内容：
[对话开始]\n"""

base_prompt_suffix_cn = """\n[对话结束]
如果你认为助手A表现更好，请直接输出A。
如果你认为助手B表现更好，请直接输出B。
不要输出任何其他内容，只需输出选项。
请输出："""

#######################Judgerbench B Prompt#############################

alignbench_judge_prompt = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better.\n
[User Question]\n{question}\n[The Start of Assistant A's Answer]\n{prediction}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{ref}\n[The End of Assistant B's Answer]"""

alpacaeval_judge_prompt = """I require a leaderboard for various large language models. I'll provide you with prompts given to these models and their corresponding outputs. Your task is to assess these responses, and select the model that produces the best output from a human perspective.

## Instruction
[[Instruction]]: {question}

## Model Outputs

Here are the unordered outputs from the models. Each output is associated with a specific model, identified by a unique model identifier.

[[model_identifier]]: "m",
[[output]]: "{prediction}"

[[model_identifier]]: "M",
[[output]]: "{ref}"

## Task

Evaluate the models based on the quality and relevance of their outputs, and select the model that generated the best output. Answer by providing the model identifier of the best model. We will use your output as the name of the best model, so make sure your output only contains one of the following model identifiers and nothing else (no quotes, no spaces, no new lines, ...): m or M.

## Best Model Identifier
"""

arenahard_judge_prompt = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.\n\nBegin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.\n\nWhen evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.\n\nThen consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.\n\nThen consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.\n\nAfter providing your explanation, you must output only one of the following choices as your final verdict with a label:\n\n1. Assistant A is significantly better: [[A>>B]]\n2. Assistant A is slightly better: [[A>B]]\n3. Tie, relatively the same: [[A=B]]\n4. Assistant B is slightly better: [[B>A]]\n5. Assistant B is significantly better: [[B>>A]]\n\nExample output: \"My final verdict is tie: [[A=B]]\".
<|User Prompt|>\n{question}\n\n<|The Start of Assistant A's Answer|>\n{prediction}\n<|The End of Assistant A's Answer|>\n\n<|The Start of Assistant B's Answer|>\n{ref}\n<|The End of Assistant B's Answer|>"""

wild_judge_prompt = """# Instruction

You are an expert evaluator. Your task is to evaluate the quality of the \
responses generated by two AI models.
We will provide you with the user query and a pair of AI-generated \
responses (Response A and Response B).
You should first read the user query and the conversation history \
carefully for analyzing the task, and then evaluate the quality of the \
responses based on and rules provided below.

# Conversation between User and AI

## History
<|begin_of_history|>

{history}

<|end_of_history|>

## Current User Query
<|begin_of_query|>

{user_query}

<|end_of_query|>

## Response A
<|begin_of_response_A|>

{prediction}

<|end_of_response_A|>

## Response B
<|begin_of_response_B|>

{ref}

<|end_of_response_B|>

# Evaluation

## Checklist

<|begin_of_checklist|>

{checklist}

<|end_of_checklist|>

Please use this checklist to guide your evaluation, but do not limit your \
assessment to the checklist.

## Rules

You should compare the above two responses based on your analysis of the \
user queries and the conversation history.
You should first write down your analysis and the checklist that you used \
for the evaluation, and then provide your assessment according to the \
checklist.
There are five choices to give your final assessment: ["A++", "A+", \
"A=B", "B+", "B++"], which correspond to the following meanings:

- `A++`: Response A is much better than Response B.
- `A+`: Response A is only slightly better than Response B.
- `A=B`: Response A and B are of the same quality. Please use this \
choice sparingly.
- `B+`: Response B is only slightly better than Response A.
- `B++`: Response B is much better than Response A.


## Output Format
First, please output your analysis for each model response, and \
then summarize your assessment to three aspects: "reason A=B", \
"reason A>B", and "reason B>A", and finally make your choice for \
the final assessment.

"""

wildbench_suffix = """Please provide your evaluation results in the following json \
format by filling in the placeholders in []:
```
{
    "analysis of A": "[analysis of Response A]",
    "analysis of B": "[analysis of Response B]",
    "reason of A=B": "[where Response A and B perform equally well]",
    "reason of A>B": "[where Response A is better than Response B]",
    "reason of B>A": "[where Response B is better than Response A]",
    "choice": "[A++ or A+ or A=B or B+ or B++]",
}
```"""

fofo_en_judge_prompt = """
I would like you to create a leaderboard that evaluates the correctness of the format of answers from various large language models. To accomplish this, you will need to analyze the text prompts given to the models and their corresponding answers. Specifically, please ensure that your evaluation outputs are properly formatted as a json string. I will provide both the prompts and the responses for this purpose.

Here is the prompt:
[[Instruction]]: {question}

Here are the outputs of the models:
[[Model Name]]: 'model',
[[Model Answer]]: '{prediction}'


Please evaluate the formatting of the model's responses by checking if they comply with the format specifications stated in the prompt. Perform a thorough format check and provide a detailed explanation for why the format is correct or incorrect. Your feedback should include the name of the model, followed by the format correctness status represented as '1' for correct and '0' for incorrect. Present your reasoning as bullet points within a single string for each model assessed. In other words, you should produce the following output:

"""

fofo_en_suffix = """```json
[
    {
        'model': <model-name>,
        'format_correctness': <correctness>,
        'reasons': <reasons-of-format-correctness>
    }
]
```

Please note that your response should be a properly formatted JSON string and should not contain any additional content. We will load it directly as a JSON string in Python."""

fofo_cn_judge_prompt = """
我希望你创建一个排行榜，用于评估来自各种大型语言模型的回答格式的正确性。为了完成这个任务，你将需要分析给模型的文本提示以及它们对应的回答。具体来说，请确保你的评估输出正确地格式化为JSON字符串。我将为此提供提示和回答。

以下是提示内容：
[[Instruction]]: {question}

以下是模型的输出结果：
[[Model Name]]: 'model',
[[Model Answer]]: '{prediction}'

请通过检查模型回答是否符合提示中声明的格式规范来评估模型回答的格式。进行彻底的格式检查，并提供格式正确或错误的详细解释。你的反馈应包括模型的名称，接着是格式正确性的状态，用'1'表示正确，'0'表示错误。将你的推理以每个评估模型的单个字符串中的 bullet 点形式呈现。换句话说，你应该生成以下输出：

"""

fofo_cn_suffix = """```json
[
    {
        'model': <模型名称>,
        'format_correctness': <正确性>,
        'reasons': <格式正确性的原因>
    }
]
```
请注意，你的回答应是一个正确格式化的JSON字符串，不应包含任何额外的内容。我们将在Python中直接将其作为JSON字符串加载。"""


def parse_conversation(conversation):
    # parse conversation into chat dialogue
    role_dict = {'user': 'HUMAN', 'assistant': 'assistant', 'HUMAN': 'HUMAN'}
    chat_round = []
    history = ''
    if len(conversation) > 0:
        for x in conversation[:-1]:
            if x['role'] == 'user' or x['role'] == 'HUMAN':
                history += 'USER: ' + x['content'] + '\n\n'
            elif x['role'] == 'assistant':
                history += 'ASSISTANT: ' + x['content'] + '\n\n'
            else:
                print(conversation)
                exit()
            chat_round.append({
                'role': role_dict[x['role']],
                'content': x['content']
            })

    last_query = conversation[-1]['content']
    chat_round.append({
        'role': role_dict[conversation[-1]['role']],
        'content': conversation[-1]['content']
    })
    chat_round.append({'role': 'assistant', 'content': ''})

    return chat_round, last_query, history


@LOAD_DATASET.register_module()
class JudgerBenchDataset(BaseDataset):

    def load(self, path: str, name: str, *args, **kwargs):

        path = get_data_path(path, local_mode=True)
        filename = osp.join(path, f'{name}.json')
        raw_data = []
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'judgerbench_A' in name:
            for item in data:
                conversation_a = item['conversation_a']
                conversation_b = item['conversation_b']
                model_a = item['model_a']
                model_b = item['model_b']
                winner = item['winner']
                category = item['category']
                turn = item['turn']
                dialogue_en, dialogue_cn = '', ''
                for itema, itemb in zip(conversation_a, conversation_b):
                    if itema['role'] == 'user':
                        dialogue_en += '### User: ' + itema['content'] + '\n'
                        dialogue_cn += '### 用户： ' + itema['content'] + '\n'
                    elif itema['role'] == 'assistant':
                        dialogue_en += '### Assistant A: ' + itema[
                            'content'] + '\n'
                        dialogue_en += '### Assistant B: ' + itemb[
                            'content'] + '\n'
                        dialogue_cn += '### 助手A： ' + itema['content'] + '\n'
                        dialogue_cn += '### 助手B： ' + itemb['content'] + '\n'
                    else:
                        raise NotImplementedError
                if '_en' in name:
                    prompt = base_prompt_prefix_en + dialogue_en + base_prompt_suffix_en
                    lan = 'en'
                elif '_cn' in name:
                    prompt = base_prompt_prefix_cn + dialogue_cn + base_prompt_suffix_cn
                    lan = 'cn'
                raw_data.append({
                    'judge_prompt': prompt,
                    'judge': {
                        'category': category,
                        'turn': turn,
                        'winner': winner,
                        'model_a': model_a,
                        'model_b': model_b,
                        'dataset_name': 'judgerbench_A',
                        'lan': lan
                    }
                })
        elif 'judgerbench_B' in name:
            for item in data:
                dataset_name = item['dataset_name']
                question = item.get('question', None)
                prediction = item['prediction']
                others = item['others']
                ref = item['others'].get('ref', None)
                if dataset_name == 'alignment_bench_v1_1':
                    judge_prompt = alignbench_judge_prompt.format(
                        question=question, prediction=prediction, ref=ref)
                elif dataset_name == 'alpaca_eval':
                    judge_prompt = alpacaeval_judge_prompt.format(
                        question=question, prediction=prediction, ref=ref)
                elif dataset_name == 'arenahard':
                    judge_prompt = arenahard_judge_prompt.format(
                        question=question, prediction=prediction, ref=ref)
                elif dataset_name == 'wildbench':
                    conversation = item['conversation']
                    checklist = item['others']['checklist']
                    chat_round, last_query, history = parse_conversation(
                        conversation)
                    judge_prompt = wild_judge_prompt.format(
                        history=history,
                        user_query=last_query,
                        prediction=prediction,
                        ref=ref,
                        checklist=checklist) + wildbench_suffix
                elif dataset_name == 'fofo_test_prompts':
                    judge_prompt = fofo_en_judge_prompt.format(
                        question=question,
                        prediction=prediction,
                    ) + fofo_en_suffix
                elif dataset_name == 'fofo_test_prompts_cn':
                    judge_prompt = fofo_cn_judge_prompt.format(
                        question=question,
                        prediction=prediction,
                    ) + fofo_cn_suffix
                raw_data.append({
                    'judge_prompt': judge_prompt,
                    'judge': {
                        'others': others,
                        'meta_judge': item['gpt4o_pred'],
                        'dataset_name': dataset_name
                    }
                })
        else:
            pass
        dataset = Dataset.from_list(raw_data)
        return dataset


class JudgerBenchEvaluator(BaseEvaluator):
    """Evaluator for followbench rule-based eval."""

    def __init__(self, num_workers=16) -> None:
        self.num_workers = num_workers

    def get_judge_result(self, judge, dataset_name):
        if dataset_name == 'alignment_bench_v1_1':
            if '[[A]]' in judge:
                return 1
            elif '[[B]]' in judge:
                return -1
            else:
                return None
        elif dataset_name == 'alpaca_eval':
            if judge[0] == 'm':
                return 1
            elif judge[0] == 'M':
                return -1
            else:
                return None
        elif dataset_name == 'fofo_test_prompts_cn' or dataset_name == 'fofo_test_prompts':
            match = re.search(r"[\"']format_correctness[\"']:\s*([0-1]+)",
                              judge)
            if match:
                score = int(match.group(1))
                return score
            else:
                return None
        elif dataset_name == 'wildbench':
            pattern = r'\"choice\": \"(.*?)\"'
            matched_result = re.findall(pattern, judge)
            if matched_result:
                if 'A++' in matched_result[0]:
                    return 2
                elif 'A+' in matched_result[0]:
                    return 1
                elif 'A=B' in matched_result[0]:
                    return 0
                elif 'B+' in matched_result[0]:
                    return -1
                elif 'B++' in matched_result[0]:
                    return -2
                else:
                    return None
            else:
                return None
        elif dataset_name == 'arenahard':
            if result := re.findall('\[\[([AB<>=]+)\]\]', judge):
                if 'A>>B' in result[0]:
                    return 2
                elif 'A>B' in result[0]:
                    return 1
                elif 'A=B' in result[0]:
                    return 0
                elif 'B>A' in result[0]:
                    return -1
                elif 'B>>A' in result[0]:
                    return -2
                else:
                    return None
            else:
                return None

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length.'}
        correlations = {}
        if references[0]['dataset_name'] == 'judgerbench_A':

            correct, total = 0, 0
            category_stats = {}

            for index, (pred, ref) in enumerate(zip(predictions, references)):
                category = ref['category']

                if ref['winner'] == 'model_a':
                    ref_winner = 'A'
                elif ref['winner'] == 'model_b':
                    ref_winner = 'B'
                else:
                    raise NotImplementedError

                if category not in category_stats:
                    category_stats[category] = {'correct': 0, 'total': 0}
                if 'A' in pred:
                    pred = 'A'
                elif 'B' in pred:
                    pred = 'B'
                is_correct = pred == ref_winner
                if is_correct:
                    category_stats[category]['correct'] += 1
                    correct += 1
                category_stats[category]['total'] += 1
                total += 1
        else:
            correct, total = 0, 0
            category_stats = {}
            models_scores = {}

            for index, (pred, ref) in enumerate(zip(predictions, references)):

                test_model, swap = ref['others']['model'], ref['others'][
                    'swap']

                dataset_name = ref['dataset_name']
                meta_judge = self.get_judge_result(ref['meta_judge'],
                                                   dataset_name)
                if meta_judge is None:
                    continue
                else:
                    model_judge = self.get_judge_result(pred, dataset_name)

                    #### Calculate absolute accuracy
                    if dataset_name not in category_stats:
                        category_stats[dataset_name] = {
                            'correct': 0,
                            'total': 0
                        }
                    is_correct = model_judge == meta_judge
                    if is_correct:
                        category_stats[dataset_name]['correct'] += 1
                        correct += 1
                    category_stats[dataset_name]['total'] += 1
                    total += 1

                    #### Calculate similarity
                    if dataset_name not in models_scores:
                        models_scores[dataset_name] = {}
                    if test_model not in models_scores[dataset_name]:
                        models_scores[dataset_name][test_model] = {
                            'meta': 0,
                            'self': 0
                        }
                    if swap:
                        models_scores[dataset_name][test_model][
                            'meta'] += -1 * meta_judge
                        if model_judge is not None:
                            models_scores[dataset_name][test_model][
                                'self'] += -1 * model_judge
                    else:
                        models_scores[dataset_name][test_model][
                            'meta'] += meta_judge
                        if model_judge is not None:
                            models_scores[dataset_name][test_model][
                                'self'] += model_judge

            for dataset, models in models_scores.items():
                meta_scores = [model['meta'] for model in models.values()]
                self_scores = [model['self'] for model in models.values()]
                correlation = np.corrcoef(meta_scores, self_scores)[0, 1]
                correlations['corr_' + dataset] = round(correlation, 3)
            average_correlation = sum(
                correlations.values()) / len(correlations)

            # 将平均值添加到字典的开始处
            correlations = {
                f'avg_corr': round(average_correlation, 3),
                **correlations
            }
        results = {'accuracy': round(correct / total, 3) if total > 0 else 0.0}

        for category, stats in category_stats.items():
            category_accuracy = round(stats['correct'] / stats['total'],
                                      3) if stats['total'] > 0 else 0.0
            results[f'accuracy_{category}'] = category_accuracy
        results.update(correlations)
        return results
