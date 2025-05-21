# flake8: noqa: E501
import json
import os.path as osp

from datasets import Dataset

from opencompass.datasets.base import BaseDataset
from opencompass.openicl import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

from .murder_mystery_solved_ex import murder_mystery_solved_ex
from .object_placements_solved_ex import object_placements_solved_ex
from .team_allocation_solved_ex import team_allocation_solved_ex
from .tree import LogicTree

DATASET_CONFIGS = {
    'murder_mysteries': {
        'file_name':
        'murder_mysteries.json',
        'ex':
        murder_mystery_solved_ex,  # write user example here
        'system_prompt':
        'You are a helpful assistant that will answer the questions given by the user.',
        'hint':
        ('Before selecting a choice, explain your reasoning step by step. '
         'The murderer needs to have a means (access to weapon), motive (reason to kill the victim), '
         'and opportunity (access to crime scene) in order to have killed the victim. '
         'Innocent suspects may have two of these proven, but not all three. '
         'An innocent suspect may be suspicious for some other reason, but they will not have all of motive, '
         'means, and opportunity established.\n\n'
         'If you believe that both suspects have motive, means, and opportunity, you should make an educated guess '
         'and pick the one for whom these are best established. If you believe that neither suspect has all '
         'three established, then choose the suspect where these are most clearly established.'
         ),
        'hint_before_question':
        False,
        'answer_index_modifier':
        1
    },
    'object_placements': {
        'file_name':
        'object_placements.json',
        'ex':
        object_placements_solved_ex,
        'skip_ablated':
        True,
        'ablation_depth_modifier':
        2,
        'system_prompt':
        'You are a helpful assistant that will answer the questions given by the user.',
        'hint':
        ('Based on this story, we want to identify where someone believes that a certain object is at the end of '
         'the story. In order to do that, you need to read the story and keep track of where they think the object '
         'is at each point. When an object is moved, the person may observe its new location if they saw it move.\n\n'
         'To see where an object ends up, they must be able to see the location that it moves to and not be too '
         'distracted by what they are doing. If they do not observe the object moving, then they will still believe '
         'it to be in the last location where they observed it.'),
        'hint_before_question':
        True,
        'answer_index_modifier':
        1
    },
    'team_allocation': {
        'file_name':
        'team_allocation.json',
        'ex':
        team_allocation_solved_ex,
        'system_prompt':
        'You are a helpful assistant that will answer the questions given by the user.',
        'hint':
        ('The story should allow you to determine how good each person is at a skill. Roughly, each person is '
         'either great, acceptable, or bad at a task. We want to find an optimal assignment of people to tasks '
         'that uses their skills as well as possible. In addition, one task will have to have two people assigned '
         'to it. The effectiveness of their teamwork (great team, acceptable team, or bad team) also impacts the '
         'overall quality of the assignment.\n\n'
         'When two people need to work on a task and one is bad at it, they don\'t necessarily benefit from the '
         'other person being good, unless they work well together.\n\n'
         'With different strengths, weaknesses, and interpersonal dynamics at play, you should allocate your team '
         'to find the single assignment to ensure that the tasks overall are completed as effectively as possible.'
         ),
        'hint_before_question':
        False,
        'answer_index_modifier':
        1
    }
}


@LOAD_DATASET.register_module()
class MusrDataset(BaseDataset):
    """MuSR.

    Args:
        path (str): path to dataset
        name (str): name of dataset
        self_consistency_n (int)
        exclude_contrastive_examples (bool): Whether to exclude contrastive examples
        reverse_contrastive_sample (bool): Whether to reverse the selection of contrastive samples
        skip_ablated (bool): Whether to skip ablated samples
        offset (int): Starting offset for the dataset
        sample_size (int): Sample size, None indicates using the entire dataset.
    """

    @staticmethod
    def load(path,
             name,
             self_consistency_n=1,
             exclude_contrastive_examples=False,
             reverse_contrastive_sample=False,
             skip_ablated=False,
             randomize=False,
             offset=0,
             sample_size=None,
             **kwargs):
        """Load the dataset and flatten fields while constructing prompts,
        taking self_consistency_n and ablations into account."""

        if name not in DATASET_CONFIGS:
            raise ValueError(
                f'Dataset name {name} not supported. Must be one of {list(DATASET_CONFIGS.keys())}'
            )

        config = DATASET_CONFIGS[name]
        path = get_data_path(path)
        file_path = osp.join(path, config['file_name'])

        with open(file_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        filtered_dataset = []
        hashes_done = []

        for example in dataset:
            if exclude_contrastive_examples and example['questions'][0].get('intermediate_data') and \
               len(example['questions'][0].get('intermediate_data')) > 0 and \
               example['questions'][0]['intermediate_data'][0].get('story_hash_id'):
                story_hash = example['questions'][0]['intermediate_data'][0][
                    'story_hash_id']
                if story_hash in hashes_done:
                    if reverse_contrastive_sample:
                        filtered_dataset.append(example)
                    else:
                        continue
                elif not reverse_contrastive_sample:
                    filtered_dataset.append(example)
                hashes_done.append(story_hash)
            else:
                filtered_dataset.append(example)

        filtered_dataset = filtered_dataset[
            offset:offset +
            min(len(filtered_dataset), sample_size) if sample_size else None]

        ablations = [
            # {'prompt': 'regular', 'name': 'regular'},
            # {'prompt': 'cot', 'name': 'cot'},
            {
                'prompt': 'cot+',
                'name': 'cot+'
            },
        ]

        # create prompts
        flattened_data = []
        for example in filtered_dataset:
            context = example['context']
            questions = example['questions']

            for question in questions:
                choices_list = question['choices']
                choices_str = '\n'.join([
                    f'{idx + 1} - {choice}'
                    for idx, choice in enumerate(choices_list)
                ])
                gold_answer = question['answer'] + config.get(
                    'answer_index_modifier', 1)

                for ablation in ablations:
                    prompt_style = ablation.get('prompt', 'cot+')
                    ablation_name = ablation.get('name', 'cot+')

                    for scidx in range(self_consistency_n):
                        ex_str = ''
                        if ablation.get('use_example') and config.get('ex'):
                            ex_str = (
                                'Here is an example of solving the task:\n\n' +
                                config.get('ex') +
                                '\n\nThis is the end of the example. The real task is below.\n\n---\n\n'
                            )

                        if prompt_style == 'regular':
                            prompt = f'{ex_str}{context}\n\n{question["question"]}\n\n' \
                                     f'Pick one of the following choices:\n{choices_str}\n\n' \
                                     'You must pick one option. Finally, the last thing you generate should be "ANSWER: (your answer here, include the choice number)"'
                        elif prompt_style == 'cot':
                            prompt = f'{ex_str}{context}\n\n{question["question"]}\n\n' \
                                     f'Pick one of the following choices:\n{choices_str}\n\n' \
                                     'You must pick one option. Explain your reasoning step by step before you answer. ' \
                                     'Finally, the last thing you generate should be "ANSWER: (your answer here, include the choice number)"'
                        elif prompt_style == 'cot+':
                            if config.get('hint_before_question'):
                                prompt = f'{ex_str}{context}\n\n{config["hint"]}\n\n{question["question"]}\n\n' \
                                         f'Pick one of the following choices:\n{choices_str}\n\n' \
                                         'You must pick one option. Explain your reasoning step by step before you answer. ' \
                                         'Finally, the last thing you generate should be "ANSWER: (your answer here, including the choice number)"'
                            else:
                                prompt = f'{ex_str}{context}\n\n{question["question"]}\n\n' \
                                         f'Pick one of the following choices:\n{choices_str}\n\n' \
                                         f'You must pick one option. {config["hint"]} Explain your reasoning step by step before you answer. ' \
                                         'Finally, the last thing you generate should be "ANSWER: (your answer here, including the choice number)"'
                        else:
                            if len(question['intermediate_trees']
                                   ) == 0 or config.get('skip_ablated', False):
                                continue

                            prompt = f'{ex_str}Answer the following questions given the list of facts per answer choice.\n\n'
                            for c, t in zip(choices_str.split('\n'),
                                            question['intermediate_trees']):
                                # extract facts from intermediate_trees
                                facts = list(
                                    set([
                                        x.value for x in
                                        LogicTree.from_json(t).get_facts(
                                            include_cs=ablation.get(
                                                'include_cs', False),
                                            include_deductions_from_level=-1,
                                            no_facts_after_depth=ablation.get(
                                                'no_facts_after_depth', 3) +
                                            config.get(
                                                'ablation_depth_modifier', 0))
                                    ]))
                                if config.get('allow_sorted_facts', True):
                                    facts = sorted(facts)
                                facts_str = '\n'.join(
                                    [f'- {fact}' for fact in facts])
                                prompt += f'Facts for Choice {c}:\n{facts_str}\n\n'
                            prompt += f'Given the list of facts per answer choice, answer the following question\n\n' \
                                      f'{question["question"]}\n\n' \
                                      f'Pick one of the following choices:\n{choices_str}\n\n' \
                                      'You must pick one option. After you have found the answer, say it in this format "ANSWER: (your answer here, include the choice number)"'

                        flattened_example = {
                            'context':
                            context,
                            'question_text':
                            question['question'],
                            'question':
                            question,
                            'answer':
                            question['answer'],
                            'choices':
                            choices_list,
                            'choices_str':
                            choices_str,
                            'intermediate_trees':
                            question.get('intermediate_trees', []),
                            'intermediate_data':
                            question.get('intermediate_data', []),
                            'prompt':
                            prompt,
                            'system_prompt':
                            config.get('system_prompt', ''),
                            'gold_answer':
                            gold_answer,
                            'scidx':
                            scidx,  # self-consistency index
                            'self_consistency_n':
                            self_consistency_n,
                            'ablation_name':
                            ablation_name,
                        }
                        flattened_data.append(flattened_example)

        dataset = Dataset.from_list(flattened_data)

        return dataset


@ICL_EVALUATORS.register_module()
class MusrEvaluator(BaseEvaluator):

    def __init__(self,
                 answer_index_modifier=1,
                 self_consistency_n=1,
                 pred_postprocessor=None):
        super().__init__(pred_postprocessor=pred_postprocessor)
        self.answer_index_modifier = answer_index_modifier
        self.self_consistency_n = self_consistency_n

    def score(self, predictions, references):
        correct = 0
        assert len(predictions) == len(
            references
        ), 'Predictions and references must have the same length!'

        total = len(predictions)

        for pred, ref in zip(predictions, references):
            if 'ANSWER:' in pred:
                answer_line = [
                    line for line in pred.split('\n') if 'ANSWER:' in line
                ]
                if answer_line:
                    answer = answer_line[0].split('ANSWER:')[-1].strip()
                    import re
                    match = re.search(r'\d+', answer)
                    if match:
                        pred_answer = int(match.group())
                        if pred_answer == ref:
                            correct += 1
        accuracy = 100 * correct / total if total > 0 else 0
        return {'accuracy': accuracy}
