# flake8: noqa: E501
import json
import os.path as osp
import re

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset

NEED_REF_CATS = ['math', 'reasoning', 'coding', 'arena-hard-200']

pair_v2 = {
    'type': 'pairwise',
    'system_prompt':
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.",
    'prompt_template':
    "[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{prediction_r1}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{prediction1_r1}\n[The End of Assistant B's Answer]",
    'description': 'Prompt for general questions',
    'category': 'general',
    'output_format': '[[A]]'
}
pair_v2_multi_turn = {
    'type': 'pairwise',
    'system_prompt':
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user questions. You should choose the assistant that follows the user's instructions and answers the user's questions better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. You should focus on who provides a better answer to the second user question. Begin your evaluation by comparing the responses of the two assistants and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.",
    'prompt_template':
    "<|The Start of Assistant A's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant A:\n{prediction_r1}\n\n### User:\n{question_2}\n\n### Assistant A:\n{prediction_r2}\n\n<|The End of Assistant A's Conversation with User|>\n\n\n<|The Start of Assistant B's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant B:\n{prediction1_r1}\n\n### User:\n{question_2}\n\n### Assistant B:\n{prediction1_r2}\n\n<|The End of Assistant B's Conversation with User|>",
    'description': 'Prompt for multi-turn general questions',
    'category': 'general',
    'output_format': '[[A]]'
}
pair_math_v1 = {
    'type': 'pairwise',
    'system_prompt':
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. Your evaluation should consider correctness and helpfulness. You will be given a reference answer, assistant A's answer, and assistant B's answer. Your job is to evaluate which assistant's answer is better. Begin your evaluation by comparing both assistants' answers with the reference answer. Identify and correct any mistakes. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.",
    'prompt_template':
    "[User Question]\n{question}\n\n[The Start of Reference Answer]\n{ref_answer_1}\n[The End of Reference Answer]\n\n[The Start of Assistant A's Answer]\n{prediction_r1}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{prediction1_r1}\n[The End of Assistant B's Answer]",
    'description': 'Prompt for math questions',
    'category': 'math',
    'output_format': '[[A]]'
}
pair_math_v1_multi_turn = {
    'type': 'pairwise',
    'system_prompt':
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user questions. Your evaluation should consider correctness and helpfulness. You will be given reference answers, the assistant A's answers, the assistant B's answers. Your job is to determine which assistant provides correct and helpful answers to the second user question. Begin your evaluation by comparing both assistants' answers with the reference answers. Identify and correct any mistakes. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.",
    'prompt_template':
    "<|The Start of Reference Answer|>\n\n### User:\n{question_1}\n\n### Reference answer:\n{ref_answer_1}\n\n### User:\n{question_2}\n\n### Reference answer:\n{ref_answer_2}\n\n<|The End of Reference Answer|>\n\n\n<|The Start of Assistant A's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant A:\n{prediction_r1}\n\n### User:\n{question_2}\n\n### Assistant A:\n{prediction_r2}\n\n<|The End of Assistant A's Conversation with User|>\n\n\n<|The Start of Assistant B's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant B:\n{prediction1_r1}\n\n### User:\n{question_2}\n\n### Assistant B:\n{prediction1_r2}\n\n<|The End of Assistant B's Conversation with User|>",
    'description': 'Prompt for multi-turn general questions',
    'category': 'general',
    'output_format': '[[A]]'
}
single_v1 = {
    'type': 'single',
    'system_prompt': 'You are a helpful assistant.',
    'prompt_template':
    "[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n[Question]\n{question}\n\n[The Start of Assistant's Answer]\n{prediction_r1}\n[The End of Assistant's Answer]",
    'description': 'Prompt for general questions',
    'category': 'general',
    'output_format': '[[rating]]'
}
single_math_v1 = {
    'type': 'single',
    'system_prompt': 'You are a helpful assistant.',
    'prompt_template':
    "[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant's answer. Begin your evaluation by comparing the assistant's answer with the reference answer. Identify and correct any mistakes. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n[Question]\n{question}\n\n[The Start of Reference Answer]\n{ref_answer_1}\n[The End of Reference Answer]\n\n[The Start of Assistant's Answer]\n{prediction_r1}\n[The End of Assistant's Answer]",
    'description': 'Prompt for general questions',
    'category': 'math',
    'output_format': '[[rating]]'
}
single_v1_multi_turn = {
    'type': 'single',
    'system_prompt':
    "Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. You evaluation should focus on the assistant's answer to the second user question. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n",
    'prompt_template':
    "<|The Start of Assistant A's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant A:\n{prediction_r1}\n\n### User:\n{question_2}\n\n### Assistant A:\n{prediction_r2}\n\n<|The End of Assistant A's Conversation with User|>",
    'description': 'Prompt for general questions',
    'category': 'general',
    'output_format': '[[rating]]'
}
single_math_v1_multi_turn = {
    'type': 'single',
    'system_prompt':
    "Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question. Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant's answer. You evaluation should focus on the assistant's answer to the second question. Begin your evaluation by comparing the assistant's answer with the reference answer. Identify and correct any mistakes. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n",
    'prompt_template':
    "<|The Start of Reference Answer|>\n\n### User:\n{question_1}\n\n### Reference answer:\n{ref_answer_1}\n\n### User:\n{question_2}\n\n### Reference answer:\n{ref_answer_2}\n\n<|The End of Reference Answer|>\n\n\n<|The Start of Assistant A's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant A:\n{prediction_r1}\n\n### User:\n{question_2}\n\n### Assistant A:\n{prediction_r2}\n\n<|The End of Assistant A's Conversation with User|>",
    'description': 'Prompt for general questions',
    'category': 'math',
    'output_format': '[[rating]]'
}


def prompt_construct(problem, multi_turn=False, judge_type='single'):
    """Return the correct pairwise judge."""
    question_1 = problem['dialogue'][0]['content']
    if multi_turn:
        question_2 = problem['dialogue'][2]['content']
        if problem['capability'] in NEED_REF_CATS:
            ref_answer_1 = problem['others']['reference'][0]
            ref_answer_2 = problem['others']['reference'][1]
            if judge_type == 'pair':
                return pair_math_v1_multi_turn[
                    'system_prompt'], pair_math_v1_multi_turn[
                        'prompt_template'].format(
                            question_1=question_1,
                            question_2=question_2,
                            ref_answer_1=ref_answer_1,
                            ref_answer_2=ref_answer_2,
                            prediction_r1='{prediction_r1}',
                            prediction_r2='{prediction_r2}',
                            prediction1_r1='{prediction1_r1}',
                            prediction1_r2='{prediction1_r2}')
            elif judge_type == 'single':
                return single_math_v1_multi_turn[
                    'system_prompt'], single_math_v1_multi_turn[
                        'prompt_template'].format(
                            question_1=question_1,
                            question_2=question_2,
                            ref_answer_1=ref_answer_1,
                            ref_answer_2=ref_answer_2,
                            prediction_r1='{prediction_r1}',
                            prediction_r2='{prediction_r2}')
        if judge_type == 'pair':
            return pair_v2_multi_turn['system_prompt'], pair_v2_multi_turn[
                'prompt_template'].format(question_1=question_1,
                                          question_2=question_2,
                                          prediction_r1='{prediction_r1}',
                                          prediction_r2='{prediction_r2}',
                                          prediction1_r1='{prediction1_r1}',
                                          prediction1_r2='{prediction1_r2}')
        elif judge_type == 'single':
            return single_v1_multi_turn['system_prompt'], single_v1_multi_turn[
                'prompt_template'].format(question_1=question_1,
                                          question_2=question_2,
                                          answer_1='{answer_1}',
                                          prediction_r1='{prediction_r1}',
                                          prediction_r2='{prediction_r2}')

    if problem['capability'] in NEED_REF_CATS:
        ref_answer_1 = problem['others']['reference'][0]
        if judge_type == 'pair':
            return pair_math_v1['system_prompt'], pair_math_v1[
                'prompt_template'].format(question=question_1,
                                          ref_answer_1=ref_answer_1,
                                          prediction_r1='{prediction_r1}',
                                          prediction1_r1='{prediction1_r1}')
        elif judge_type == 'single':
            return single_math_v1['system_prompt'], single_math_v1[
                'prompt_template'].format(question=question_1,
                                          ref_answer_1=ref_answer_1,
                                          prediction_r1='{prediction_r1}')
    else:
        if judge_type == 'pair':
            return pair_v2['system_prompt'], pair_v2['prompt_template'].format(
                question=question_1,
                prediction_r1='{prediction_r1}',
                prediction1_r1='{prediction1_r1}')
        elif judge_type == 'single':
            return single_v1['system_prompt'], single_v1[
                'prompt_template'].format(question=question_1,
                                          prediction_r1='{prediction_r1}')


@LOAD_DATASET.register_module()
class MTBenchDataset(BaseDataset):

    def load(self,
             path: str,
             name: str,
             judge_type='single',
             multi_turn=True,
             *args,
             **kwargs):
        path = get_data_path(path, local_mode=True)
        filename = osp.join(path, f'{name}.json')
        dataset = DatasetDict()
        raw_data = []
        with open(filename, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            for problem in json_data:
                if 'dialogue' in problem:
                    system_prompt, prompt_template = prompt_construct(
                        problem, multi_turn, judge_type)
                    dialogue = problem['dialogue']
                    capability = problem['capability']
                    others = problem['others']
                    others['round'] = int(len(dialogue) / 2)
                    user_contents = [
                        item['content'] for item in dialogue
                        if item['role'] == 'user'
                    ]
                    question = ' '.join(user_contents)
                    others['question'] = question
                    raw_data.append({
                        'dialogue': dialogue,
                        'capability': capability,
                        'system_prompt': system_prompt,
                        'prompt_template': prompt_template,
                        'others': others,
                        'judge': {
                            'capability': capability,
                            'others': others,
                        }
                    })
        dataset = Dataset.from_list(raw_data)
        return dataset
