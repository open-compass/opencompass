import importlib
from typing import Any

DEFAULT_IMAGE_TOKEN = '<image>'
DEFAULT_IMAGE_PATCH_TOKEN = '<im_patch>'
DEFAULT_IM_START_TOKEN = '<im_start>'
DEFAULT_IM_END_TOKEN = '<im_end>'


class LLaVAMMBenchPromptConstructor:
    """Prompt constructor for LLaVA on MMBench.

    Args:
        conv_templates (Any): Conversation class to build prompt.
        conv_mode (str): Version control args for different version of LLaVA.
        mm_use_im_start_end (bool):
            Config arg. Use start and end token when build prompt or not.
    """

    def __init__(self, conv_templates: Any, conv_mode: str,
                 mm_use_im_start_end: bool) -> None:
        self.conv_templates = conv_templates
        self.conv_mode = conv_mode
        self.mm_use_im_start_end = mm_use_im_start_end
        conversation = importlib.import_module('llava.conversation')
        self.SeparatorStyle = conversation.SeparatorStyle

    def __call__(self, inputs: dict) -> tuple:
        """Construct prompt.

        Args:
            inputs (dict): Input data containing images and data_samples.

        Returns:
            tuple: A tuple containing prompt, images and data_samples.
        """
        data_samples = inputs['data_samples']
        assert len(data_samples) == 1
        question = data_samples[0].get('question')
        options = data_samples[0].get('options')
        context = data_samples[0].get('context')
        if context is not None:
            prompt = context + ' ' + question + ' ' + options
        else:
            prompt = question + ' ' + options
        if self.mm_use_im_start_end:
            prompt = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN +
                      DEFAULT_IM_END_TOKEN + '\n' + prompt)
        else:
            prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt  # noqa

        conv = self.conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        output_prompt = conv.get_prompt()

        stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2  # noqa

        return output_prompt, stop_str
