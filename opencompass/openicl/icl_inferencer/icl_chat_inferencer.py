"""Chat Inferencer."""
import os
import os.path as osp
from typing import List, Optional, Union

import mmengine
from mmengine import is_list_of
from tqdm import tqdm

from opencompass.models import APITemplateParser as _APITemplateParser
from opencompass.models import BaseModel
from opencompass.models import LMTemplateParser as _LMTemplateParser
from opencompass.registry import ICL_INFERENCERS
from opencompass.utils.prompt import PromptList

from ..icl_prompt_template import PromptTemplate
from ..icl_retriever import BaseRetriever
from ..utils.logging import get_logger
from .icl_base_inferencer import BaseInferencer, dump_results_dict

logger = get_logger(__name__)


def promptlist_to_openai(prompt: Union[str, PromptList]):
    output = []
    if isinstance(prompt, str):
        return [dict(role='user', content=prompt)]

    for item in prompt:
        if 'section' in item:
            continue
        if isinstance(item, str) and item:
            output.append(dict(role='user', content=item))
        elif item['role'] == 'SYSTEM':
            output.append(dict(role='system', content=item['prompt']))
        elif item['role'] == 'HUMAN':
            output.append(dict(role='user', content=item['prompt']))
        elif item['role'] == 'BOT':
            output.append(dict(role='assistant', content=item['prompt']))
    return output


class LMTemplateParser:
    """LMTemplateParser accepts OpenAI format dialog inputs."""

    def __init__(self, meta_template: Optional[dict] = None):
        self.meta_template = meta_template
        self.roles = {}
        role_mapping = {
            'SYSTEM': 'system',
            'HUMAN': 'user',
            'BOT': 'assistant',
        }
        if meta_template:
            for item in meta_template.get('round', []):
                role = role_mapping.get(item['role'], item['role'])
                self.roles[role] = item.copy()
            for item in meta_template.get('reserved_roles', []):
                role = role_mapping.get(item['role'], item['role'])
                self.roles[role] = item.copy()

    def parse_template(self, chat: List[dict], mode='gen') -> str:
        if is_list_of(chat, list):
            # Handle batch inputs
            return [self.parse_template(item) for item in chat]

        assert is_list_of(chat, dict)
        prompt = ''
        if self.roles:
            for dialog in chat:
                role_cfg = self.roles.get(dialog['role'], {})
                prompt += (role_cfg.get('begin') or '')
                prompt += (dialog.get('content') or '')
                prompt += (role_cfg.get('end') or '')
            prompt += (self.roles['assistant'].get('begin') or '')
        else:
            # in case the model does not have any meta template
            last_sep = ''
            for item in chat:
                prompt += last_sep + (item.get('content') or '')
                last_sep = '\n'
        return prompt


class APITemplateParser:
    """APITemplateParser accepts OpenAI format dialog inputs."""

    def __init__(self, meta_template: Optional[dict] = None):
        self.meta_template = meta_template
        self.roles = {}
        role_mapping = {
            'SYSTEM': 'system',
            'HUMAN': 'user',
            'BOT': 'assistant',
        }
        if meta_template:
            for item in meta_template.get('round', []):
                role = role_mapping.get(item['role'], item['role'])
                self.roles[role] = item.copy()
            for item in meta_template.get('reserved_roles', []):
                role = role_mapping.get(item['role'], item['role'])
                self.roles[role] = item.copy()
        else:
            self.roles = dict(
                system=dict(api_role='SYSTEM'),
                user=dict(api_role='HUMAN'),
                assistant=dict(api_role='BOT', generate=True),
            )

    def parse_template(self, chat: List[dict], mode='gen') -> str:
        if is_list_of(chat, list):
            # Handle batch inputs
            return [self.parse_template(item) for item in chat]

        assert is_list_of(chat, dict)
        prompt = []
        for dialog in chat:
            if dialog['role'] in self.roles:
                role = self.roles[dialog['role']]['api_role']
            else:
                role = dialog['role']
            prompt.append(dict(role=role, prompt=dialog.get('content') or ''))
        return PromptList(prompt)


class ChatOutputHandler:

    def __init__(self) -> None:
        self.results_dict = {}

    def write_to_json(self, save_dir: str, filename: str):
        """Dump the result to a json file."""
        dump_results_dict(self.results_dict, osp.join(save_dir, filename))

    def save_results(self,
                     origin_prompt: list,
                     prediction: str,
                     idx: int,
                     gold: str = None):
        result_dict = {}
        if gold:
            result_dict['gold'] = gold
        result_dict.update({
            'prediction': prediction,
            'origin_prompt': origin_prompt,
        })
        self.results_dict[str(idx)] = result_dict

    def save_multiround_results(self,
                                origin_prompt: list,
                                prediction: str,
                                idx: int,
                                gold: str = None):
        result_dict = self.results_dict.get(str(idx), {
            'gold': [],
            'prediction': [],
            'origin_prompt': [],
        })
        result_dict['gold'].append(gold)
        result_dict['prediction'].append(prediction)
        result_dict['origin_prompt'].append(origin_prompt)
        self.results_dict[str(idx)] = result_dict


@ICL_INFERENCERS.register_module()
class ChatInferencer(BaseInferencer):
    HandlerType = ChatOutputHandler

    def __init__(
            self,
            model,
            output_json_filepath: Optional[str] = './icl_inference_output',
            output_json_filename: Optional[str] = 'predictions',
            save_every: Optional[int] = 1,
            infer_mode: str = 'last',
            max_out_len: int = 512,
            **kwargs) -> None:
        super().__init__(
            model=model,
            output_json_filename=output_json_filename,
            output_json_filepath=output_json_filepath,
            **kwargs,
        )
        assert infer_mode in ['last', 'every', 'every_with_gt']
        self.infer_mode = infer_mode
        self.model: BaseModel
        self._set_meta_template(self.model)

        if self.model.is_api and save_every is None:
            save_every = 1
        self.save_every = save_every
        self.dialogue_mode = False
        self.max_out_len = max_out_len

    def _set_meta_template(self, model):
        origin = model.template_parser
        if isinstance(origin, _APITemplateParser):
            model.template_parser = APITemplateParser(origin.meta_template)
        if isinstance(origin, _LMTemplateParser):
            model.template_parser = LMTemplateParser(origin.meta_template)

    def inference(self,
                  retriever: BaseRetriever,
                  ice_template: Optional[PromptTemplate] = None,
                  prompt_template: Optional[PromptTemplate] = None,
                  output_json_filepath: Optional[str] = None,
                  output_json_filename: Optional[str] = None) -> dict:
        # 1. Preparation for output logs
        output_handler = self.HandlerType()

        if output_json_filepath is None:
            output_json_filepath = self.output_json_filepath
        if output_json_filename is None:
            output_json_filename = self.output_json_filename

        # 2. Get results of retrieval process
        ice_idx_list = retriever.retrieve()

        # 3. Generate prompts for testing input
        chat_list = self.get_chat_list(
            ice_idx_list,
            retriever,
            prompt_template=prompt_template,
        )

        # Create tmp json file for saving intermediate results and future
        # resuming
        index = 0
        tmp_json_filepath = os.path.join(output_json_filepath,
                                         'tmp_' + output_json_filename)
        if osp.exists(tmp_json_filepath):
            # TODO: move resume to output handler
            try:
                tmp_result_dict = mmengine.load(tmp_json_filepath)
            except Exception:
                pass
            else:
                output_handler.results_dict = tmp_result_dict
                index = len(tmp_result_dict)

        # 4. Wrap prompts with Dataloader
        dataloader = self.get_dataloader(chat_list[index:], batch_size=1)

        # 5. Inference for prompts in each batch
        logger.info('Starting inference process...')
        for datum in tqdm(dataloader, disable=not self.is_main_process):
            chat = datum[0]

            if self.infer_mode == 'last':
                self.infer_last(chat, index, output_handler)
            elif self.infer_mode == 'every':
                self.infer_every(chat, index, output_handler)
            elif self.infer_mode == 'every_with_gt':
                self.infer_every_with_gt(chat, index, output_handler)
            index += 1

            # Save intermediate results
            if (self.save_every is not None and index % self.save_every == 0
                    and self.is_main_process):
                output_handler.write_to_json(output_json_filepath,
                                             'tmp_' + output_json_filename)

        # 4. Output
        if self.is_main_process:
            os.makedirs(output_json_filepath, exist_ok=True)
            output_handler.write_to_json(output_json_filepath,
                                         output_json_filename)
            if osp.exists(tmp_json_filepath):
                os.remove(tmp_json_filepath)

        return output_handler.results_dict

    def get_chat_list(self,
                      ice_idx_list: List[List[int]],
                      retriever: BaseRetriever,
                      prompt_template: Optional[PromptTemplate] = None):
        prompt_list = []
        input_columns = retriever.dataset_reader.input_columns
        output_column = retriever.dataset_reader.output_column

        def chat_from_entry(entry):
            if prompt_template is None and len(input_columns) == 1:
                # Directly use the input column as the user input
                user = entry.get(input_columns[0])
                assistant = entry.get(output_column, '')
                return [
                    dict(role='user', content=user),
                    dict(role='assistant', content=assistant),
                ]
            elif prompt_template is not None:
                # Use prompt template to generate chat history
                chat = promptlist_to_openai(
                    prompt_template.generate_item(entry))
                gold = entry.get(output_column, '')
                if chat[-1]['role'] != 'assistant':
                    chat.append(dict(role='assistant', content=gold))
                return chat
            else:
                raise ValueError()

        for idx, ice_idx in enumerate(ice_idx_list):
            # NOTE: The in-context examples won't be used by now.

            item = {
                k: v
                for k, v in retriever.test_ds[idx].items()
                if k in input_columns or k == output_column
            }
            if all(isinstance(value, str) for value in item.values()):
                # Every column is a single string
                chat = chat_from_entry(item)
            elif all(is_list_of(value, str) for value in item.values()):
                # Every column is a list of string for multi-round chat
                entries = [dict(zip(item, v)) for v in zip(*item.values())]
                chat = sum((chat_from_entry(entry) for entry in entries), [])
            elif len(input_columns) == 1 and is_list_of(
                    item[input_columns[0]], dict):
                # Single input column and it's already a chat.
                chat = item[input_columns[0]]
            elif 'dialogue' in input_columns:
                chat = item['dialogue']
                self.dialogue_mode = True
            else:
                raise ValueError('Cannot construct chat from the dataset.')

            prompt_list.append(chat)
        return prompt_list

    def infer_last(self, chat: List[dict], index: int, output_handler):
        assistant_indices = [
            i for i, item in enumerate(chat) if item['role'] == 'assistant'
        ]

        history = chat[:assistant_indices[-1]]
        output = self.model.generate_from_template(
            [history], max_out_len=self.max_out_len)[0]
        output_handler.save_results(
            origin_prompt=history,
            prediction=output,
            idx=index,
            gold=chat[assistant_indices[-1]]['content'],
        )

    def infer_every(self, chat: List[dict], index: int, output_handler):
        assistant_indices = [
            i for i, item in enumerate(chat) if item['role'] == 'assistant'
        ]
        index_copy = index

        for i in assistant_indices:
            history = chat[:i]
            output = self.model.generate_from_template(
                [history], max_out_len=self.max_out_len)[0]
            chat[i]['content'] = output
            if not self.dialogue_mode:
                output_handler.save_multiround_results(
                    origin_prompt=history[-1]['content'],
                    prediction=output,
                    idx=index,
                    gold=chat[i]['content'],
                )
                index += 1
        if self.dialogue_mode:
            # dialogue mode for subjective evaluation
            assert len(chat) % 2 == 0
            round_num = int(len(chat) / 2)
            preds_list = []
            for i in range(round_num):
                temp_dict = {
                    'round': i + 1,
                    'user': chat[i * 2]['content'],
                    'assistant': chat[i * 2 + 1]['content']
                }
                preds_list.append(temp_dict)
            output_handler.save_results(
                origin_prompt=None,
                prediction=preds_list,
                idx=index_copy,
                gold=None,
            )

    def infer_every_with_gt(self, chat: List[dict], index: int,
                            output_handler):
        assistant_indices = [
            i for i, item in enumerate(chat) if item['role'] == 'assistant'
        ]

        for i in assistant_indices:
            history = chat[:i]
            output = self.model.generate_from_template(
                [history], max_out_len=self.max_out_len)[0]
            output_handler.save_multiround_results(
                origin_prompt=history[-1]['content'],
                prediction=output,
                idx=index,
                gold=chat[i]['content'],
            )
            index += 1
