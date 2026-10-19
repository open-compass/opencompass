"""Agent Inferencer."""
import os.path as osp
import types
from typing import List

from opencompass.models.lagent import LagentAgent
from opencompass.registry import ICL_INFERENCERS

from ..utils.logging import get_logger
from .icl_base_inferencer import dump_results_dict
from .icl_chat_inferencer import ChatInferencer

logger = get_logger(__name__)


class AgentInferencerOutputHandler:

    def __init__(self) -> None:
        self.results_dict = {}

    def write_to_json(self, save_dir: str, filename: str):
        """Dump the result to a json file."""
        dump_results_dict(self.results_dict, osp.join(save_dir, filename))

    def save_results(self,
                     origin_prompt: list,
                     prediction: str,
                     steps: list,
                     idx: int,
                     gold: str = None):
        result_dict = {}
        if gold:
            result_dict['gold'] = gold
        result_dict.update({
            'prediction': prediction,
            'origin_prompt': origin_prompt,
            'steps': steps,
        })
        self.results_dict[str(idx)] = result_dict

    def save_multiround_results(self,
                                origin_prompt: list,
                                prediction: str,
                                steps: list,
                                idx: int,
                                gold: str = None):
        result_dict = self.results_dict.get(str(idx), {
            'gold': [],
            'prediction': [],
            'origin_prompt': [],
            'steps': [],
        })
        result_dict['gold'].append(gold)
        result_dict['prediction'].append(prediction)
        result_dict['origin_prompt'].append(origin_prompt)
        result_dict['steps'].append(steps)
        self.results_dict[str(idx)] = result_dict


def model_adapter(model):
    """Modify the generate method to accept and return single item."""
    if getattr(model, '_generate_is_wrapped', False):
        # Avoid wrap twice.
        return model

    origin_generate = model.generate

    def generate(self, inputs, *args, **kwargs):
        return origin_generate([inputs], *args, **kwargs)[0]

    model.generate = types.MethodType(generate, model)
    setattr(model, '_generate_is_wrapped', True)
    return model


@ICL_INFERENCERS.register_module()
class AgentInferencer(ChatInferencer):
    HandlerType = AgentInferencerOutputHandler

    def __init__(self, model, **kwargs) -> None:
        model.agent._llm = model_adapter(model.agent._llm)
        super().__init__(model, **kwargs)
        self.model: LagentAgent

    def infer_last(self, chat: List[dict], index: int, output_handler):
        assistant_indices = [
            i for i, item in enumerate(chat) if item['role'] == 'assistant'
        ]

        user_idx = assistant_indices[-1] - 1
        self.model.set_history(chat[:user_idx])
        answer, steps, _ = self.model.chat(chat[user_idx]['content'])
        output_handler.save_results(
            origin_prompt=chat[user_idx]['content'],
            prediction=answer,
            steps=steps,
            idx=index,
            gold=chat[assistant_indices[-1]]['content'],
        )
        self.model.reset()

    def infer_every(self, chat: List[dict], index: int, output_handler):
        assistant_indices = [
            i for i, item in enumerate(chat) if item['role'] == 'assistant'
        ]

        history = chat[:assistant_indices[0] - 1]
        for i in assistant_indices:
            answer, steps, inner_steps = self.model.chat(
                chat[i - 1]['content'], history)
            history += inner_steps
            output_handler.save_multiround_results(
                origin_prompt=chat[i - 1]['content'],
                prediction=answer,
                steps=steps,
                idx=index,
                gold=chat[i]['content'],
            )
        self.model.reset()

    def infer_every_with_gt(self, chat: List[dict], index: int,
                            output_handler):
        assistant_indices = [
            i for i, item in enumerate(chat) if item['role'] == 'assistant'
        ]

        history = chat[:assistant_indices[0] - 1]
        prev_idx = 0
        for i in assistant_indices:
            for j in range(prev_idx, i - 1):
                if chat[j]['role'] == 'assistant':
                    history += self.model.gt_response(chat[j]['content'])
                elif chat[j]['role'] == 'user':
                    history += [chat[j]]
            self.model.set_history(history)
            answer, steps, _ = self.model.chat(chat[i - 1]['content'])
            output_handler.save_multiround_results(
                origin_prompt=chat[i - 1]['content'],
                prediction=answer,
                steps=steps,
                idx=index,
                gold=chat[i]['content'],
            )
            history += [chat[i - 1]]
            prev_idx = i
        self.model.reset()
