import copy

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.clip import CLIPVisionConfig

from .mpt.configuration_mpt import MPTConfig

logger = logging.get_logger(__name__)


class OtterConfig(PretrainedConfig):
    r"""
    [`OtterConfig`] is the configuration class to store the configuration of a [`OtterForConditionalGeneration`]. It is
    used to instantiate a Otter model according to the specified arguments, defining the vision model and language model configs. Instantiating a configuration with the defaults will yield a similar configuration to
    that of the Otter architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`PretrainedConfig`].
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize any [`PretrainedConfig`].
        cross_attn_every_n_layers (`int`, *optional*, defaults to 4):
            The number of cross-attention layers adding after each transformer layer.

        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import (
    ...     PretrainedConfig,
    ...     OPTConfig,
    ...     OtterConfig,
    ...     OtterForConditionalGeneration,
    ... )

    >>> # Initializing a OtterConfig with luodian/otter-9b-hf style configuration
    >>> configuration = OtterConfig()

    >>> # Initializing a OtterForConditionalGeneration (with random weights) from the Salesforce/Otter-opt-2.7b style configuration
    >>> model = OtterForConditionalGeneration(configuration)
    ```"""
    model_type = "otter"
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        cross_attn_every_n_layers: int = 4,
        use_media_placement_augmentation: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. initializing the vision config with default values.")

        if text_config is None:
            text_config = {}
            logger.info("text_config is None. Initializing the text config with default values.")

        self.vision_config = CLIPVisionConfig(**vision_config)
        if "architectures" in text_config.keys() and text_config["architectures"] != None:
            if text_config["architectures"][0] == "MPTForCausalLM":
                self.text_config = MPTConfig(**text_config)
            elif text_config["architectures"][0] == "RWForCausalLM":
                self.text_config = RWConfig(**text_config)
            elif text_config["architectures"][0] == "LlamaForCausalLM":
                self.text_config = CONFIG_MAPPING[text_config.pop("model_type")](**text_config)
            else:
                import pdb

                pdb.set_trace()
        else:
            self.text_config = CONFIG_MAPPING[text_config.pop("model_type")](**text_config)

        self.cross_attn_every_n_layers = cross_attn_every_n_layers
        self.use_media_placement_augmentation = use_media_placement_augmentation

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["vision_config"] = self.vision_config.to_dict()
        output["text_config"] = self.text_config.to_dict()
        output["model_type"] = self.__class__.model_type
        output["cross_attn_every_n_layers"] = self.cross_attn_every_n_layers
        output["use_media_placement_augmentation"] = self.use_media_placement_augmentation
        return output
