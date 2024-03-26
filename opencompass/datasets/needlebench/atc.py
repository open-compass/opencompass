# flake8: noqa
import json
import random

from datasets import Dataset

from opencompass.datasets.base import BaseDataset
from opencompass.registry import LOAD_DATASET


@LOAD_DATASET.register_module()
class NeedleBenchATCDataset(BaseDataset):

    @staticmethod
    def load(
        path,
        num_needles: int,
        language: str,
        repeats: int,
    ):
        data = {'prompt': [], 'answer': []}

        with open(path, 'r', encoding='utf-8') as file:
            names_data = json.load(file)

        all_names = names_data[language].split(',')

        for _ in range(repeats):
            names = random.sample(all_names, num_needles)
            if language == 'Chinese':

                relationship_terms = [
                    '父亲', '母亲', '爸爸', '妈妈', '爷爷', '奶奶', '姥姥', '姥爷', '外公', '外婆'
                ]

                relationship_templates = [
                    '{A}是{B}的{relationship}。',
                    '{B}的{relationship}是{A}。',
                    '{A}作为{B}的{relationship}，对{B}的成长有重要影响。',
                    '{A}不仅是{B}的{relationship}，还是{B}的榜样。',
                    '{B}是{A}所生的孩子。',
                    '{A}对{B}来说，不只是一个{relationship}，还是一个朋友。',
                    '{A}在{B}的生命中扮演着{relationship}的角色。',
                    '{B}把{A}视为其{relationship}。',
                ]
            elif language == 'English':

                relationship_terms = [
                    'father', 'mother', 'dad', 'mom', 'grandfather',
                    'grandmother', 'maternal grandmother',
                    'maternal grandfather', 'paternal grandfather',
                    'paternal grandmother'
                ]

                relationship_templates = [
                    "{A} is {B}'s {relationship}.",
                    "{B}'s {relationship} is {A}.",
                    ("{A}, as {B}'s {relationship}, "
                     "has a significant impact on {B}'s upbringing."),
                    ("{A} is not only {B}'s {relationship} "
                     "but also {B}'s role model."),
                    '{B} is the child of {A}.',
                    ('For {B}, {A} is not just a {relationship}, '
                     'but also a friend.'),
                    ("{A} plays the role of {B}'s {relationship} "
                     "in {B}'s life."),
                    '{B} considers {A} as their {relationship}.',
                ]

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
                prompt = (f"""
在上面提供的打乱的家族关系文本中，'{last_person}'的能够向上追溯到的最年长的亲人是谁？
例如：
例子1.如果张强的父亲是马克，除此以外提供的文本中没有更多关于亲属关系的信息，那么在提供的文本中张强能够向上追溯到的最年长的亲人就是马克。
例子2.如果李明的姥姥是张红，而张红的父亲是张强，除此以外提供的文本中没有更多关于亲属关系的信息，那么在提供的文本中李明能够向上追溯到的最年长的亲人就是张强。
例子3.如果小明是张红的曾孙女，张红的祖母是王华，王华的父亲是王刚，除此以外提供的文本中没有更多关于亲属关系的信息，那么小明能够向上追溯到的最年长的亲人就是王刚。
""")
            elif language == 'English':
                prompt = (f"""
Given the scrambled family relationships described above, who is the eldest relative that '{last_person}' can trace back to in the context?
For example:
Example 1: If Zhang Qiang's father is Mark, and no further information about familial relationships is provided in the text, then the oldest relative Zhang Qiang can trace back to in the provided text is Mark.
Example 2: If Li Ming's grandmother is Zhang Hong, and Zhang Hong's father is Zhang Qiang, and no further information about familial relationships is provided in the text, then the oldest relative Li Ming can trace back to in the provided text is Zhang Qiang.
Example 3: If Xiao Ming is Zhang Hong's great-granddaughter, Zhang Hong's grandmother is Wang Hua, and Wang Hua's father is Wang Gang, and no further information about familial relationships is provided in the text, then the oldest relative Xiao Ming can trace back to in the provided text is Wang Gang."""
                          )
            else:
                prompt = 'Language not supported.'
                raise Exception('Unsupported language specified. '
                                "Please choose either 'Chinese' or 'English'.")

            # Combine story and prompt
            shuffled_story_with_prompt = shuffled_story + ' ' + prompt

            data['prompt'].append(shuffled_story_with_prompt)
            data['answer'].append(names[0] + '*' + names[0])

        dataset = Dataset.from_dict({
            'prompt': data['prompt'],
            'answer': data['answer'],
        })
        return dataset


@LOAD_DATASET.register_module()
class NeedleBenchATCOrderedDataset(BaseDataset):

    @staticmethod
    def load(
        path,
        num_needles: int,
        language: str,
        repeats: int,
    ):
        data = {'prompt': [], 'answer': []}

        with open(path, 'r', encoding='utf-8') as file:
            names_data = json.load(file)

        all_names = names_data[language].split(',')

        for _ in range(repeats):
            names = random.sample(all_names, num_needles)
            if language == 'Chinese':

                relationship_terms = [
                    '父亲', '母亲', '爸爸', '妈妈', '爷爷', '奶奶', '姥姥', '姥爷', '外公', '外婆'
                ]

                relationship_templates = [
                    '{A}是{B}的{relationship}。',
                    '{B}的{relationship}是{A}。',
                    '{A}作为{B}的{relationship}，对{B}的成长有重要影响。',
                    '{A}不仅是{B}的{relationship}，还是{B}的榜样。',
                    '{B}是{A}所生的孩子。',
                    '{A}对{B}来说，不只是一个{relationship}，还是一个朋友。',
                    '{A}在{B}的生命中扮演着{relationship}的角色。',
                    '{B}把{A}视为其{relationship}。',
                ]
            elif language == 'English':

                relationship_terms = [
                    'father', 'mother', 'dad', 'mom', 'grandfather',
                    'grandmother', 'maternal grandmother',
                    'maternal grandfather', 'paternal grandfather',
                    'paternal grandmother'
                ]

                relationship_templates = [
                    "{A} is {B}'s {relationship}.",
                    "{B}'s {relationship} is {A}.",
                    ("{A}, as {B}'s {relationship}, "
                     "has a significant impact on {B}'s upbringing."),
                    ("{A} is not only {B}'s {relationship} "
                     "but also {B}'s role model."),
                    '{B} is the child of {A}.',
                    ('For {B}, {A} is not just a {relationship}, '
                     'but also a friend.'),
                    ("{A} plays the role of {B}'s {relationship} "
                     "in {B}'s life."),
                    '{B} considers {A} as their {relationship}.',
                ]

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

            # Joining the shuffled fragments back into a string
            shuffled_story = ''.join(family_story_fragments)

            last_person = names[-1]

            # Generating the prompt based on the language
            if language == 'Chinese':
                prompt = (f"""
在上面提供的打乱的家族关系文本中，'{last_person}'的能够向上追溯到的最年长的亲人是谁？
例如：
例子1.如果张强的父亲是马克，除此以外提供的文本中没有更多关于亲属关系的信息，那么在提供的文本中张强能够向上追溯到的最年长的亲人就是马克。
例子2.如果李明的姥姥是张红，而张红的父亲是张强，除此以外提供的文本中没有更多关于亲属关系的信息，那么在提供的文本中李明能够向上追溯到的最年长的亲人就是张强。
例子3.如果小明是张红的曾孙女，张红的祖母是王华，王华的父亲是王刚，除此以外提供的文本中没有更多关于亲属关系的信息，那么小明能够向上追溯到的最年长的亲人就是王刚。
""")
            elif language == 'English':
                prompt = (f"""
Given the scrambled family relationships described above, who is the eldest relative that '{last_person}' can trace back to in the context?
For example:
Example 1: If Zhang Qiang's father is Mark, and no further information about familial relationships is provided in the text, then the oldest relative Zhang Qiang can trace back to in the provided text is Mark.
Example 2: If Li Ming's grandmother is Zhang Hong, and Zhang Hong's father is Zhang Qiang, and no further information about familial relationships is provided in the text, then the oldest relative Li Ming can trace back to in the provided text is Zhang Qiang.
Example 3: If Xiao Ming is Zhang Hong's great-granddaughter, Zhang Hong's grandmother is Wang Hua, and Wang Hua's father is Wang Gang, and no further information about familial relationships is provided in the text, then the oldest relative Xiao Ming can trace back to in the provided text is Wang Gang."""
                          )
            else:
                prompt = 'Language not supported.'
                raise Exception('Unsupported language specified. '
                                "Please choose either 'Chinese' or 'English'.")

            # Combine story and prompt
            shuffled_story_with_prompt = shuffled_story + ' ' + prompt

            data['prompt'].append(shuffled_story_with_prompt)
            data['answer'].append(names[0] + '*' + names[0])

        dataset = Dataset.from_dict({
            'prompt': data['prompt'],
            'answer': data['answer'],
        })
        return dataset
