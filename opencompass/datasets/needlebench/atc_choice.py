# flake8: noqa
import copy
import json
import random

from datasets import Dataset

from opencompass.registry import LOAD_DATASET

from ..base import BaseDataset


def get_number(options):
    result_string = ''
    for i, option in enumerate(options, start=ord('A')):
        result_string += f'{chr(i)}. {option}\n'
    return result_string


def get_circular_example(entry, id):
    """For given example, generate four circular examples."""
    # Only 4 options is supported for current circular eval.
    circular_patterns = ['ABCD', 'BCDA', 'CDAB', 'DABC']
    data = []
    for c in circular_patterns:
        line = copy.deepcopy(entry)
        options = []
        for i in range(4):
            options.append(line['options'][ord(c[i]) - ord('A')])
        line['options'] = options
        line['answer'] = {
            c[0]: 'A',
            c[1]: 'B',
            c[2]: 'C',
            c[3]: 'D'
        }[line['answer']]
        line['answer'] = str(id) + '--' + line['answer'] + '--' + c
        line['question'] = line['question'].strip() + '\n' + get_number(
            line['options'])
        data.append(line)

    return data


@LOAD_DATASET.register_module()
class NeedleBenchATCDataset(BaseDataset):

    @staticmethod
    def load(path: str,
             num_needles: int,
             language: str,
             repeats: int,
             with_circular: bool = True):
        """NeedleBenthATC Dataset.

        Args:
            path (str): Path of the needlebench dataset.
            name (str): Name of the target subset.
            with_circular (bool): Whether to create circular dataset for
                single choice question. Defaults to True.
        """
        data = []
        entry = {}

        with open(path, 'r', encoding='utf-8') as file:
            names_data = json.load(file)

        all_names = names_data[language].split(',')

        for id in range(repeats):
            random.seed(id)
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
在上面提供的打乱的家族关系文本中，'{last_person}'的能够向上追溯到的最年长的亲人是谁？""")
            elif language == 'English':
                prompt = (f"""
Given the scrambled family relationships described above, who is the eldest relative that '{last_person}' can trace back to in the context?"""
                          )
            else:
                prompt = 'Language not supported.'
                raise Exception('Unsupported language specified. '
                                "Please choose either 'Chinese' or 'English'.")

            # Combine story and prompt
            shuffled_story_with_prompt = shuffled_story + ' ' + prompt

            entry['question'] = shuffled_story_with_prompt
            if len(names) < 4:
                additional_names_needed = max(4 - len(names), 0)
                additional_names = random.sample(
                    [name for name in all_names if name not in names],
                    additional_names_needed)
                names.extend(additional_names)

            entry['options'] = names[0:4]
            entry['answer'] = 'A'
            # print(entry)
            data.extend(get_circular_example(entry, id))
        dataset = Dataset.from_list(data)
        return dataset
