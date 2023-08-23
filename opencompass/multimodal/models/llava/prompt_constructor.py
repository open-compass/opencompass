import importlib

DEFAULT_IMAGE_TOKEN = '<image>'
DEFAULT_IMAGE_PATCH_TOKEN = '<im_patch>'
DEFAULT_IM_START_TOKEN = '<im_start>'
DEFAULT_IM_END_TOKEN = '<im_end>'


class LLaVABasePromptConstructor:
    """Base prompt constructor for LLaVA.

    Args:
        conv_mode (str): Version control args for different version of LLaVA.
        mm_use_im_start_end (bool):
            Config arg. Use start and end token when build prompt or not.
    """

    def __init__(self, conv_mode: str, mm_use_im_start_end: bool) -> None:
        conversation = importlib.import_module('llava.conversation')
        self.conv_templates = conversation.conv_templates
        self.conv_mode = conv_mode
        self.mm_use_im_start_end = mm_use_im_start_end
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
        prompt = self._build_prompt(data_samples[0])
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

    def _build_prompt(self, data_sample):
        return ''


class LLaVAMMBenchPromptConstructor(LLaVABasePromptConstructor):
    """MMBench prompt constructor for LLaVA.

    Args:
        conv_mode (str): Version control args for different version of LLaVA.
        mm_use_im_start_end (bool):
            Config arg. Use start and end token when build prompt or not.
    """

    def __init__(self, conv_mode: str, mm_use_im_start_end: bool) -> None:
        super().__init__(conv_mode, mm_use_im_start_end)

    def _build_prompt(self, data_sample):
        question = data_sample.get('question')
        options = data_sample.get('options')
        context = data_sample.get('context')
        if context is not None:
            prompt = context + ' ' + question + ' ' + options
        else:
            prompt = question + ' ' + options
        return prompt
