# flake8: noqa: E501
import json
import os
import random

import tiktoken
from datasets import Dataset
from huggingface_hub import hf_hub_download

from opencompass.datasets.base import BaseDataset
from opencompass.datasets.needlebench_v2.atc import (
    relationship_templates_en, relationship_templates_zh_CN,
    relationship_terms_en, relationship_terms_zh_CN)
from opencompass.registry import LOAD_DATASET


def get_random_needles(counter, file_path, num_needles, language):
    with open(file_path, 'r', encoding='utf-8') as file:
        names_data = json.load(file)

    all_names = names_data[language].split(',')

    random.seed(counter)
    names = random.sample(all_names, num_needles)

    if language == 'Chinese':
        relationship_terms = relationship_terms_zh_CN
        relationship_templates = relationship_templates_zh_CN
    elif language == 'English':
        relationship_terms = relationship_terms_en
        relationship_templates = relationship_templates_en
    else:
        raise ValueError(f"Unsupported language '{language}' specified.")

    def generate_chain_family_story(names, templates, relationship_terms):
        story = ''
        for i in range(len(names) - 1):
            template = random.choice(templates)
            relation_term = random.choice(relationship_terms)
            relation = template.format(A=names[i],
                                       B=names[i + 1],
                                       relationship=relation_term)
            story += f'{relation}*'
        return story

    chain_story = generate_chain_family_story(names, relationship_templates,
                                              relationship_terms)

    # Splitting the chain_story into a list of fragments
    family_story_fragments = chain_story.split('*')

    # Removing the empty string from the list
    family_story_fragments = [
        fragment for fragment in family_story_fragments if fragment
    ]

    # Shuffling the list of fragments
    random.shuffle(family_story_fragments)

    last_person = names[-1]

    # Generating the retrieval question based on the language
    if language == 'Chinese':
        retrieval_question = f"在上面提供的文本中，'{last_person}'的能够向上追溯到的最年长的亲人是谁？"
    elif language == 'English':
        retrieval_question = f"Given the context described above, who is the eldest relative that '{last_person}' can trace back to in the context?"

    # Returning the story, answer, and retrieval question
    return {
        'needles': family_story_fragments,
        'answer': names[0],
        'retrieval_question': retrieval_question,
        'last_person': last_person
    }


@LOAD_DATASET.register_module()
class NeedleBenchMultiDataset(BaseDataset):

    @staticmethod
    def load(
        path: str,  # depreciated
        length: int,
        depth: int,
        tokenizer_model: str,
        file_list: 'list[str]',
        num_repeats_per_file: int,
        length_buffer: int,
        language: str,
        needle_file_name: str,
        num_needles: int,
        diff: int,
        quesiton_position: str = 'End',
    ):
        data = {'prompt': [], 'answer': []}
        tokenizer = tiktoken.encoding_for_model(tokenizer_model)

        def _generate_context(tokens_context, depth_percent, needles):
            tokens_needle = [
                _get_tokens_from_context(needle) for needle in needles
            ]
            insertion_points = []
            total_length = len(tokens_context)

            for i, needle_tokens in enumerate(tokens_needle):
                if i == 0:
                    insertion_point = int(total_length * (depth_percent / 100))
                else:
                    insertion_point = int(insertion_points[i - 1] +
                                          len(tokens_needle[i - 1]) +
                                          total_length * (diff / 100))
                insertion_point = min(
                    insertion_point,
                    total_length + sum(len(tn) for tn in tokens_needle[:i]))
                insertion_points.append(insertion_point)

            for i, needle_tokens in enumerate(tokens_needle):
                tokens_context = tokens_context[:insertion_points[i]] \
                    + needle_tokens + tokens_context[insertion_points[i]:]
                for j in range(i + 1, len(insertion_points)):
                    insertion_points[j] += len(needle_tokens)

            new_context = _decode_tokens(tokens_context)
            return new_context

        def _get_tokens_from_context(context):
            if isinstance(context, list):
                return [tokenizer.encode(item) for item in context]
            else:
                return tokenizer.encode(context)

        def _decode_tokens(tokens):
            return tokenizer.decode(tokens)

        def _generate_prompt(context, retrieval_question, last_person):

            if language == 'Chinese':
                if quesiton_position == 'End':
                    prompt = f'''这是一个长文本能力的测试，你需要首先阅读下面的长文档，然后根据文档中的信息回答最后的问题。
长文档的内容如下

<文档>
{context}
</文档>

根据文档中的信息，现在请问：{retrieval_question}

例如：
例子1.如果张强的父亲是马克，除此以外提供的文本中没有更多关于亲属关系的信息，那么在提供的文本中张强能够向上追溯到的最年长的亲人就是马克。
例子2.如果李明的姥姥是张红，而张红的父亲是张强，除此以外提供的文本中没有更多关于亲属关系的信息，那么在提供的文本中李明能够向上追溯到的最年长的亲人就是张强。
例子3.如果小明是张红的曾孙女，张红的祖母是王华，王华的父亲是王刚，除此以外提供的文本中没有更多关于亲属关系的信息，那么小明能够向上追溯到的最年长的亲人就是王刚。

注意：
1. 你不必纠结这个测试中的人名的性别关系，例如，一个通常被视为女性化的名字仍然可以是其他人的父亲，我们的重点是谁更年长。
2. 忽略这个测试中的姓氏遗传问题，例如，李明仍然可能是王鹏的亲生父亲，我们只关注谁更年长，不必纠结孩子是否应该继承父亲或母亲的性别。
3. 在回答的最后，将你的答案放在\\boxed{{}}中，例如：“所以{last_person}能向上追溯到的最年长的亲人就是\\boxed{{（你的答案）}}”

'''
                elif quesiton_position == 'Start':
                    prompt = f'''这是一个长文本能力的测试，你需要首先阅读下面的问题，然后根据最后长文档中的信息回答下面的问题。
现在请问：{retrieval_question}

例如：
例子1.如果张强的父亲是马克，除此以外提供的文本中没有更多关于亲属关系的信息，那么在提供的文本中张强能够向上追溯到的最年长的亲人就是马克。
例子2.如果李明的姥姥是张红，而张红的父亲是张强，除此以外提供的文本中没有更多关于亲属关系的信息，那么在提供的文本中李明能够向上追溯到的最年长的亲人就是张强。
例子3.如果小明是张红的曾孙女，张红的祖母是王华，王华的父亲是王刚，除此以外提供的文本中没有更多关于亲属关系的信息，那么小明能够向上追溯到的最年长的亲人就是王刚。

注意：
1. 你不必纠结这个测试中的人名的性别关系，例如，一个通常被视为女性化的名字仍然可以是其他人的父亲，我们的重点是谁更年长。
2. 忽略这个测试中的姓氏遗传问题，例如，李明仍然可能是王鹏的亲生父亲，我们只关注谁更年长，不必纠结孩子是否应该继承父亲或母亲的性别。
3. 在回答的最后，将你的答案放在\\boxed{{}}中，例如：“所以{last_person}能向上追溯到的最年长的亲人就是\\boxed{{（你的答案）}}”

长文档内容的如下

<文档>
{context}
</文档>

'''
                else:
                    raise ValueError('Unsupported quesiton_position. '
                                     'Position must be "End" or "Start".')
            elif language == 'English':
                if quesiton_position == 'End':
                    prompt = f'''This is a test of long-text capability. You need to first read the long document below, and then answer the final question based on the information in the document.
The content of the long document is as follows

<Document>
{context}
</Document>

Based on the information in the document, now please answer: {retrieval_question}

For example:
Example 1: If James Hill's father is Jasmine Lane, and no further information about familial relationships is provided in the text, then the oldest relative James Hill can trace back to in the provided text is \\boxed{{Jasmine Lane}}.
Example 2: If Andrew Williams's grandmother is Dan Newton, and Dan Newton's father is James Hill, and no further information about familial relationships is provided in the text, then the oldest relative Andrew Williams can trace back to in the provided text is \\boxed{{James Hill}}.
Example 3: If Jeff White's father is Kevin Le, Dan Newton's grandmother is Jeff White, and Jeff White's father is Kevin Le, and Shelley Mills is Dan Newton's great-granddaughter, and no further information about familial relationships is provided in the text, then the oldest relative Shelley Mills can trace back to in the provided text is \\boxed{{Kevin Le}}.

Notes:
1. You do not need to worry about the gender consistency of names in this test. For example, a name that is typically considered feminine can still be the father of another person. Our primary focus is on who is older.
2. Ignore surname inheritance issues. For instance, Andrew Williams could still be the biological father of Christopher Baker. We only care about who is older and do not need to consider whether a child should inherit the father's or mother's surname.
3. At the end of your response, remember to put your final answer within \\boxed{{}}. For example: "So the oldest relative '{last_person}' can trace back to in the provided text is \\boxed{{(your answer here)}}."

'''
                elif quesiton_position == 'Start':
                    prompt = f'''This is a test of long-text capability. You need to first read the question below, and then answer it based on the information in the long document that follows.
Now please answer: {retrieval_question}

For example:
Example 1: If James Hill's father is Jasmine Lane, and no further information about familial relationships is provided in the text, then the oldest relative James Hill can trace back to in the provided text is \\boxed{{Jasmine Lane}}.
Example 2: If Andrew Williams's grandmother is Dan Newton, and Dan Newton's father is James Hill, and no further information about familial relationships is provided in the text, then the oldest relative Andrew Williams can trace back to in the provided text is \\boxed{{James Hill}}.
Example 3: If Jeff White's father is Kevin Le, Dan Newton's grandmother is Jeff White, and Jeff White's father is Kevin Le, and Shelley Mills is Dan Newton's great-granddaughter, and no further information about familial relationships is provided in the text, then the oldest relative Shelley Mills can trace back to in the provided text is \\boxed{{Kevin Le}}.

Notes:
1. You do not need to worry about the gender consistency of names in this test. For example, a name that is typically considered feminine can still be the father of another person. Our primary focus is on who is older.
2. Ignore surname inheritance issues. For instance, Andrew Williams could still be the biological father of Christopher Baker. We only care about who is older and do not need to consider whether a child should inherit the father's or mother's surname.
3. At the end of your response, remember to put your final answer within \\boxed{{}}. For example: "So the oldest relative '{last_person}' can trace back to in the provided text is \\boxed{{(your answer here)}}."

The content of the long document is as follows

<Document>
{context}
</Document>

'''
                else:
                    raise ValueError(
                        f'Unsupported quesiton_position {quesiton_position}. '
                        'Position must be "End" or "Start".')
            else:
                raise ValueError(f"Language '{language}' is not supported.")

            return prompt

        repo_id = 'opencompass/NeedleBench'
        file_names = [
            'PaulGrahamEssays.jsonl', 'names.json', 'zh_finance.jsonl',
            'zh_game.jsonl', 'zh_general.jsonl', 'zh_government.jsonl',
            'zh_movie.jsonl', 'zh_tech.jsonl'
        ]
        downloaded_files = []
        base_file_path = ''
        for file_name in file_names:
            file_path = hf_hub_download(repo_id=repo_id,
                                        filename=file_name,
                                        repo_type='dataset')
            downloaded_files.append(file_path)
            base_file_path = '/'.join(file_path.split('/')[:-1])

        needle_file_path = os.path.join(base_file_path, needle_file_name)
        for file_path in downloaded_files:
            if file_path.split('/')[-1] not in file_list:
                continue

            with open(file_path, 'r', encoding='utf-8') as f:
                lines_bak = [json.loads(line.strip()) for line in f]
            lines = lines_bak.copy()
            for counter in range(num_repeats_per_file):
                random.seed(counter)
                random.shuffle(lines)
                random_needle_data = get_random_needles(
                    counter, needle_file_path, num_needles + 1, language)
                last_person = random_needle_data['last_person']
                needles = [
                    '\n' + needle + '\n'
                    for needle in random_needle_data['needles']
                ]
                answer = random_needle_data['answer']
                keyword = answer
                retrieval_question = random_needle_data['retrieval_question']
                context_length = length - length_buffer
                target_length_per_record = context_length - \
                    sum(len(tokens) for tokens
                        in _get_tokens_from_context(needles))
                target_length_per_record = max(target_length_per_record, 0)
                accumulated_tokens = []
                for line in lines:
                    tokens_current_line = _get_tokens_from_context(
                        line['text'])
                    accumulated_tokens.extend(tokens_current_line)

                    if len(accumulated_tokens) >= target_length_per_record:
                        break

                processed_text = _generate_context(
                    accumulated_tokens[:target_length_per_record], depth,
                    needles)

                processed_prompt = _generate_prompt(processed_text,
                                                    retrieval_question,
                                                    last_person)

                data['prompt'].append(processed_prompt)
                data['answer'].append(keyword)

        dataset = Dataset.from_dict({
            'prompt': data['prompt'],
            'answer': data['answer'],
        })
        return dataset
