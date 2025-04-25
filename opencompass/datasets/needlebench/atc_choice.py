# flake8: noqa
import copy
import json
import os
import random

from datasets import Dataset
import numpy as np

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset
from .atc import relationship_terms_zh_CN, relationship_templates_zh_CN, relationship_terms_en, relationship_templates_en

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
    def load(
        path: str,
        file_name: str,
        num_needles: int,
        language: str,
        repeats: int,
        with_circular: bool = True,
    ):
        """NeedleBenthATC Dataset.

        Args:
            path (str): Path of the needlebench dataset.
            name (str): Name of the target subset.
            with_circular (bool): Whether to create circular dataset for
                single choice question. Defaults to True.
        """
        data = []
        entry = {}
        path = get_data_path(path)
        if os.environ.get('DATASET_SOURCE') == 'HF':
            from huggingface_hub import snapshot_download

            path = snapshot_download(repo_id=path, repo_type='dataset')
        file_path = os.path.join(path, file_name)

        with open(file_path, 'r', encoding='utf-8') as file:
            names_data = json.load(file)

        all_names = names_data[language].split(',')

        for id in range(repeats):
            random.seed(id)
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
                prompt = f"""
下面是对你的多步推理能力的测试，这个测试叫做祖先追溯测试，我们会模拟不同人的家庭亲属关系，你的任务是在其中不断推理，直到找到最年长的祖先。

注意：
1. 你不必纠结这个测试中的人名的性别关系，例如，一个通常被视为女性化的名字仍然可以是其他人的父亲，我们的重点是谁更年长。
2. 忽略这个测试中的姓氏遗传问题，例如，李明仍然可能是王鹏的亲生父亲，我们只关注谁更年长，不必纠结孩子是否应该继承父亲或母亲的性别。
3. 在回答的最后，用“所以答案是A/B/C/D”的格式回答正确答案(你只能选择一个选项)。

现在，打乱的家族关系文本如下：
{shuffled_story}

在上面提供的打乱的家族关系文本中，'{last_person}'的能够向上追溯到的最年长的亲人是谁？

例如：
例子1.如果张强的父亲是马克，除此以外提供的文本中没有更多关于亲属关系的信息，那么在提供的文本中张强能够向上追溯到的最年长的亲人就是马克。
例子2.如果李明的姥姥是张红，而张红的父亲是张强，除此以外提供的文本中没有更多关于亲属关系的信息，那么在提供的文本中李明能够向上追溯到的最年长的亲人就是张强。
例子3.如果小明是张红的曾孙女，张红的祖母是王华，王华的父亲是王刚，除此以外提供的文本中没有更多关于亲属关系的信息，那么小明能够向上追溯到的最年长的亲人就是王刚。
"""
            elif language == 'English':
                prompt = f"""
Here is a test for multi-step reasoning ability called the Ancestral Trace Challenge. In this test, we will simulate different people’s familial relationships, and your task is to continuously reason through them until you identify the eldest ancestor.

Notes:
1. You do not need to worry about the gender consistency of names in this test. For example, a name that is typically considered feminine can still be the father of another person. Our primary focus is on who is older.
2. Ignore surname inheritance issues. For instance, Li Ming could still be the biological father of Wang Peng. We only care about who is older and do not need to consider whether a child should inherit the father’s or mother’s surname.
3. At the end of your response, state the correct answer in the format: “So the answer is A/B/C/D.”(You must choose only one option.)

Now, the scrambled family relationships are provided below:
{shuffled_story}

Given the scrambled family relationships described above, who is the eldest relative that '{last_person}' can trace back to in the context?

For example:
Example 1: If Zhang Qiang's father is Mark, and no further information about familial relationships is provided in the text, then the oldest relative Zhang Qiang can trace back to in the provided text is Mark.
Example 2: If Li Ming's grandmother is Zhang Hong, and Zhang Hong's father is Zhang Qiang, and no further information about familial relationships is provided in the text, then the oldest relative Li Ming can trace back to in the provided text is Zhang Qiang.
Example 3: If Xiao Ming is Zhang Hong's great-granddaughter, Zhang Hong's grandmother is Wang Hua, and Wang Hua's father is Wang Gang, and no further information about familial relationships is provided in the text, then the oldest relative Xiao Ming can trace back to in the provided text is Wang Gang."""
            else:
                prompt = 'Language not supported.'
                raise Exception('Unsupported language specified. '
                                "Please choose either 'Chinese' or 'English'.")

            # Combine story and prompt
            shuffled_story_with_prompt = prompt

            entry['question'] = shuffled_story_with_prompt
            if len(names) < 4:
                additional_names_needed = max(4 - len(names), 0)
                additional_names = random.sample(
                    [name for name in all_names if name not in names],
                    additional_names_needed,
                )
                names.extend(additional_names)

            num_samples = 3  
            if len(names) > 1:
                indices = np.linspace(1, len(names) - 1, num_samples, dtype=int)  # Generate evenly spaced indices
                sampled_names = [names[i] for i in indices]  # Select corresponding elements
                entry['options'] = names[:1] + sampled_names
            else:
                entry['options'] = names  # Return directly if only one element
            assert len(entry['options']) == 4
            entry['answer'] = 'A'
            # print(entry)
            data.extend(get_circular_example(entry, id))
        dataset = Dataset.from_list(data)
        return dataset
