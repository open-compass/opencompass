# flake8: noqa
import json
import os
import random
import re

from datasets import Dataset

from opencompass.datasets.base import BaseDataset
from opencompass.datasets.math import extract_boxed_answer
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import (ICL_EVALUATORS, LOAD_DATASET,
                                  TEXT_POSTPROCESSORS)
from opencompass.utils import get_data_path

relationship_templates_zh_CN = [
    '{A}是{B}的{relationship}。',
    '{B}的{relationship}是{A}。',
    '{A}作为{B}的{relationship}，对{B}的成长有重要影响。',
    '{A}不仅是{B}的{relationship}，还是{B}的榜样。',
    '{B}是{A}所生的孩子。',
    '{A}对{B}来说，不只是一个{relationship}，还是一个朋友。',
]

relationship_terms_zh_CN = [
    '父亲',
    '母亲',
    '爸爸',
    '妈妈',
    '爷爷',
    '奶奶',
    '姥姥',
    '姥爷',
    '外公',
    '外婆',
]

relationship_terms_en = [
    'father',
    'mother',
    'dad',
    'mom',
    'grandfather',
    'grandmother',
    'maternal grandmother',
    'maternal grandfather',
    'paternal grandfather',
    'paternal grandmother',
]

relationship_templates_en = [
    "{A} is {B}'s {relationship}.",
    "{B}'s {relationship} is {A}.",
    ("{A}, as {B}'s {relationship}, "
     "has a significant impact on {B}'s upbringing."),
    ("{A} is not only {B}'s {relationship} "
     "but also {B}'s role model."),
    '{B} is the child of {A}.',
    ('For {B}, {A} is not just a {relationship}, '
     'but also a friend.'),
    'For {B}, {A} is more than just a {relationship}; {A} is a lifelong mentor of {B}.',
]

shuffled_story_with_prompt_zh_CN = """下面是对你的多步推理能力的测试，这个测试叫做祖先追溯测试，我们会模拟不同人的家庭亲属关系，你的任务是在其中不断推理，直到找到最年长的祖先。
                
例如：
例子1.如果张强的父亲是马克，除此以外提供的文本中没有更多关于亲属关系的信息，那么在提供的文本中张强能够向上追溯到的最年长的亲人就是马克。
例子2.如果李明的姥姥是张红，而张红的父亲是张强，除此以外提供的文本中没有更多关于亲属关系的信息，那么在提供的文本中李明能够向上追溯到的最年长的亲人就是张强。
例子3.如果小明是张红的曾孙女，张红的祖母是王华，王华的父亲是王刚，除此以外提供的文本中没有更多关于亲属关系的信息，那么小明能够向上追溯到的最年长的亲人就是王刚。

注意：
1. 你不必纠结这个测试中的人名的性别关系，例如，一个通常被视为女性化的名字仍然可以是其他人的父亲，我们的重点是谁更年长。
2. 忽略这个测试中的姓氏遗传问题，例如，李明仍然可能是王鹏的亲生父亲，我们只关注谁更年长，不必纠结孩子是否应该继承父亲或母亲的性别。
3. 在回答的最后，将你的答案放在\\boxed{{}}中，例如："所以{last_person}能向上追溯到的最年长的亲人就是\\boxed{{某人（你的答案）}}"

现在，打乱的家族关系文本如下：
{shuffled_story}

在上面提供的打乱的家族关系文本中，'{last_person}'的能够向上追溯到的最年长的亲人是谁？
"""

shuffled_story_with_prompt_en = """Here is a test for multi-step reasoning ability called the Ancestral Trace Challenge. In this test, we will simulate different people's familial relationships, and your task is to continuously reason through them until you identify the eldest ancestor.

For example:
Example 1: If James Hill's father is Jasmine Lane, and no further information about familial relationships is provided in the text, then the oldest relative James Hill can trace back to in the provided text is \\boxed{{Jasmine Lane}}.
Example 2: If Andrew Williams's grandmother is Dan Newton, and Dan Newton's father is James Hill, and no further information about familial relationships is provided in the text, then the oldest relative Andrew Williams can trace back to in the provided text is \\boxed{{James Hill}}.
Example 3: If Jeff White's father is Kevin Le, Dan Newton's grandmother is Jeff White, and Jeff White's father is Kevin Le, and Shelley Mills is Dan Newton's great-granddaughter, and no further information about familial relationships is provided in the text, then the oldest relative Shelley Mills can trace back to in the provided text is \\boxed{{Kevin Le}}.

Notes:
1. You do not need to worry about the gender consistency of names in this test. For example, a name that is typically considered feminine can still be the father of another person. Our primary focus is on who is older.
2. Ignore surname inheritance issues. For instance, Andrew Williams could still be the biological father of Christopher Baker. We only care about who is older and do not need to consider whether a child should inherit the father's or mother's surname.
3. At the end of your response, remember to put your final answer within \\boxed{{}}. For example: "So the oldest relative '{last_person}' can trace back to in the provided text is \\boxed{{somebody (your answer here)}}."

Now, the scrambled family relationships are provided below:
{shuffled_story}

Given the scrambled family relationships described above, who is the eldest relative that '{last_person}' can trace back to in the context?
"""


@LOAD_DATASET.register_module()
class NeedleBenchATCDataset(BaseDataset):

    @staticmethod
    def load(
        path,
        file_name: str,
        num_needles: int,
        language: str,
        repeats: int,
    ):
        data = {'prompt': [], 'answer': []}
        path = get_data_path(path)
        if os.environ.get('DATASET_SOURCE') == 'HF':
            from huggingface_hub import snapshot_download

            path = snapshot_download(repo_id=path, repo_type='dataset')
        file_path = os.path.join(path, file_name)

        with open(file_path, 'r', encoding='utf-8') as file:
            names_data = json.load(file)

        all_names = names_data[language].split(',')

        for i in range(repeats):
            # 使用固定种子来保持样本稳定性
            seed = i
            random.seed(seed)

            names = random.sample(all_names, num_needles)
            if language == 'Chinese':
                relationship_terms = relationship_terms_zh_CN
                relationship_templates = relationship_templates_zh_CN
            elif language == 'English':
                relationship_terms = relationship_terms_en
                relationship_templates = relationship_templates_en

            def generate_chain_family_story(names, templates,
                                            relationship_terms):
                story = ''
                for i in range(len(names) - 1):
                    template = random.choice(templates)
                    relation_term = random.choice(relationship_terms)
                    relation = template.format(A=names[i],
                                               B=names[i + 1],
                                               relationship=relation_term)
                    story += f'{relation}*'
                return story

            chain_story = generate_chain_family_story(names,
                                                      relationship_templates,
                                                      relationship_terms)

            # Splitting the chain_story into a list of fragments
            family_story_fragments = chain_story.split('*')

            # Shuffling the list of fragments
            random.shuffle(family_story_fragments)

            # Joining the shuffled fragments back into a string
            shuffled_story = ''.join(family_story_fragments)

            last_person = names[-1]

            # Generating the prompt based on the language
            if language == 'Chinese':
                shuffled_story_with_prompt = shuffled_story_with_prompt_zh_CN.format(
                    shuffled_story=shuffled_story, last_person=last_person)
            elif language == 'English':
                shuffled_story_with_prompt = shuffled_story_with_prompt_en.format(
                    shuffled_story=shuffled_story, last_person=last_person)
            else:
                prompt = 'Language not supported.'
                raise Exception('Unsupported language specified. '
                                "Please choose either 'Chinese' or 'English'.")

            data['prompt'].append(shuffled_story_with_prompt)
            data['answer'].append(names[0])

        dataset = Dataset.from_dict({
            'prompt': data['prompt'],
            'answer': data['answer'],
        })
        return dataset


def clean_atc_answer(text: str) -> str:
    """Clean answer format specifically for QwQ-32B-Preview model.

    Args:
        text: Raw prediction text

    Returns:
        Standardized name format after cleaning
    """
    if not text or text == 'None':
        return 'None'

    # Remove LaTeX commands but keep content
    text = re.sub(r'\\text\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\boxed\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\[\[\]]', '', text)

    # Remove extra backslashes
    text = text.replace('\\\\', '').replace('\\', '')

    # Handle extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove quotes
    text = text.replace('"', '').replace("'", '')
    # Remove tildes (波浪符号)
    text = text.replace('~', ' ')

    return text


@TEXT_POSTPROCESSORS.register_module('needlebench_atc_postprocess_v2')
def needlebench_atc_postprocess_v2(text: str) -> str:

    cand_ans = extract_boxed_answer(text, strip_double_curly_brace=True)

    if cand_ans:
        return clean_atc_answer(cand_ans)
    return 'None'


@ICL_EVALUATORS.register_module('needlebench_atc_evaluator')
class NeedleBenchATCEvaluator(BaseEvaluator):

    def score(self, predictions, gold):
        if len(predictions) != len(gold):
            return {'error': 'predictions and gold have different lengths'}

        correct_count = 0
        details = []

        for prediction, reference in zip(predictions, gold):
            reference_name = reference
            if prediction.strip() == reference_name.strip():
                correct_count += 1

            detail = {
                'pred': prediction,
                'answer': reference_name,
                'correct': prediction.strip() == reference_name.strip()
            }
            details.append(detail)

        accuracy = (correct_count /
                    len(predictions)) * 100 if predictions else 0
        result = {'score': accuracy, 'details': details}
        return result
