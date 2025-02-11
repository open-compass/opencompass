# flake8: noqa
# yapf: disable

# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
from typing import Dict, List, Optional, Union

from absl import flags

import opencompass.datasets.xIFEval.instructions_registry as instructions_registry

# _INPUT_DATA = flags.DEFINE_string('input_data',
#                                   None,
#                                   'path to input data',
#                                   required=True)

# _INPUT_RESPONSE_DATA = flags.DEFINE_string('input_response_data',
#                                            None,
#                                            'path to input response data',
#                                            required=False)

# _OUTPUT_DIR = flags.DEFINE_string(
#     'output_dir',
#     None,
#     'Output directory for inference and eval results.',
#     required=True,
# )


@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: List[str]
    prompt: str
    kwargs: List[Dict[str, Optional[Union[str, int]]]]
    lang: str


@dataclasses.dataclass
class OutputExample:
    instruction_id_list: List[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: List[bool]


# def test_instruction_following_strict(
#     inp,
#     response,
# ):
#     """Tests response to see if instrutions are followed."""
#     instruction_list = inp.instruction_id_list
#     is_following_list = []

#     for index, instruction_id in enumerate(instruction_list):
#         instruction_cls = instructions_registry.INSTRUCTION_DICT[
#             instruction_id]
#         instruction = instruction_cls(instruction_id)
#         instruction.build_description(**inp.kwargs[index])
#         args = instruction.get_instruction_args()
#         if args and 'prompt' in args:
#             instruction.build_description(prompt=inp.prompt)

#         if response.strip() and instruction.check_following(response):
#             is_following_list.append(True)
#         else:
#             is_following_list.append(False)

#     return OutputExample(
#         instruction_id_list=inp.instruction_id_list,
#         prompt=inp.prompt,
#         response=response,
#         follow_all_instructions=all(is_following_list),
#         follow_instruction_list=is_following_list,
#     )


# def test_instruction_following_loose(
#     inp,
#     response,
# ):
#     """Tests response for an upper bound for following instructions."""
#     r = response.split('\n')
#     response_remove_first = '\n'.join(r[1:]).strip()
#     response_remove_last = '\n'.join(r[:-1]).strip()
#     response_remove_both = '\n'.join(r[1:-1]).strip()
#     revised_response = response.replace('*', '')
#     revised_response_remove_first = response_remove_first.replace('*', '')
#     revised_response_remove_last = response_remove_last.replace('*', '')
#     revised_response_remove_both = response_remove_both.replace('*', '')
#     all_responses = [
#         response,
#         revised_response,
#         response_remove_first,
#         response_remove_last,
#         response_remove_both,
#         revised_response_remove_first,
#         revised_response_remove_last,
#         revised_response_remove_both,
#     ]
#     instruction_list = inp.instruction_id_list
#     is_following_list = []

#     for index, instruction_id in enumerate(instruction_list):
#         instruction_cls = instructions_registry.INSTRUCTION_DICT[
#             instruction_id]
#         instruction = instruction_cls(instruction_id)

#         instruction.build_description(**inp.kwargs[index])
#         args = instruction.get_instruction_args()
#         if args and 'prompt' in args:
#             instruction.build_description(prompt=inp.prompt)

#         is_following = False
#         for r in all_responses:
#             if r.strip() and instruction.check_following(r):
#                 is_following = True
#                 break

#         is_following_list.append(is_following)

#     return OutputExample(
#         instruction_id_list=inp.instruction_id_list,
#         prompt=inp.prompt,
#         response=response,
#         follow_all_instructions=all(is_following_list),
#         follow_instruction_list=is_following_list,
#     )


def test_instruction_following_strict(
    inp,
    response,
):
    """Tests response to see if instructions are followed."""
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id, inp.lang)

        # Remove None values from kwargs to avoid unexpected keyword argument errors in build_description method.
        kwargs = {k: v for k, v in inp.kwargs[index].items() if v}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        if response.strip() and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def test_instruction_following_loose(
    inp,
    response,
):
    """Tests response for an upper bound for following instructions."""
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id, inp.lang)

        # Remove None values from kwargs to avoid unexpected keyword argument errors in build_description method.
        kwargs = {k: v for k, v in inp.kwargs[index].items() if v}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        is_following = False
        for r in all_responses:
            if r.strip() and instruction.check_following(r):
                is_following = True
                break

        is_following_list.append(is_following)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def process_results(doc, results):
    inp = InputExample(
        key=doc["key"],
        instruction_id_list=doc["instruction_id_list"],
        prompt=doc["prompt"],
        kwargs=doc["kwargs"],
        lang=doc.get("lang", "en")
    )
    response = results[0]

    out_strict = test_instruction_following_strict(inp, response)
    out_loose = test_instruction_following_loose(inp, response)

    return {
        "prompt_level_strict_acc": out_strict.follow_all_instructions,
        "inst_level_strict_acc": out_strict.follow_instruction_list,
        "prompt_level_loose_acc": out_loose.follow_all_instructions,
        "inst_level_loose_acc": out_loose.follow_instruction_list,
    }


def agg_inst_level_acc(items):
    flat_items = [item for sublist in items for item in sublist]
    inst_level_acc = sum(flat_items) / len(flat_items)
    return inst_level_acc