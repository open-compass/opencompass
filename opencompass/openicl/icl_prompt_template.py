"""Prompt Template."""
import copy
from typing import Dict, Hashable, List, Optional, Union

from opencompass.registry import ICL_PROMPT_TEMPLATES
from opencompass.utils.prompt import PromptList, safe_format
from opencompass.utils.types import _check_type_list

PromptType = Union[PromptList, str]


@ICL_PROMPT_TEMPLATES.register_module()
class PromptTemplate:
    """In-context Learning Prompt Template Class This class represents a
    template that guides the generation of prompts in the retrieval or
    inference process.

    Attributes:
        template (:obj:`Dict` or :obj:`str`): A custom template dictionary or
            string. If a dictionary, the keys of the dictionary represent the
            values of the output_column, and the values represent the
            corresponding generated statement. If a string, it represents a
            string template.
        ice_token(:obj:`str`, optional): A string that represents the specific
            token mapping from in-context examples. None if you want to use
            this template only to generate in-context examples, otherwise it
            can be used to generate the final prompt that is fed into the PLM.
            The ice_token will be invisible when generating in-context
            examples.
    """

    def __init__(
        self,
        template: Union[Dict, str],
        ice_token: Optional[str] = None,
        sep_token: Optional[str] = None,
    ) -> None:
        self.template = template
        assert isinstance(self.template, (str, Dict))
        self.ice_token = _check_type_list(ice_token, [None, str])
        self.sep_token = _check_type_list(sep_token, [None, str])
        # A sign used to distinguish the prompt type
        self.prompt_type = 'origin'
        self._check_template_legacy()

    def _check_template_legacy(self):
        if isinstance(self.template, Dict):
            # Check if it's the label-prompt type or just a meta prompt type
            ctr = sum(key in self.template
                      for key in ('begin', 'round', 'end'))
            self.prompt_type = 'meta' if ctr == len(
                self.template.keys()) else 'origin'

            # Check if token exists in values of tp_dict
            for tp_dict_val in self.template.values():
                if not isinstance(tp_dict_val, (str, list, dict)):
                    raise TypeError(
                        'dictionary of template expects a str, list or a '
                        f"dict, but got '{tp_dict_val}'")
                if isinstance(
                        tp_dict_val, str
                ) and self.ice_token and self.ice_token not in tp_dict_val:
                    raise LookupError(
                        f"'{self.ice_token}' not in '{tp_dict_val}'")

        if isinstance(self.template, str):
            if self.ice_token and self.ice_token not in self.template:
                raise LookupError(
                    f"'{self.ice_token}' not in '{self.template}'")

    def generate_ice_item(self, entry: Dict, label: Hashable) -> PromptType:
        """Generate in-context example based on the provided :obj:`entry` data.

        Args:
            entry (:obj:`Dict`): A piece of data to be used for generating the
                in-context example.
            label (:obj:`Hashable`): The value of the output field.

        Returns:
            PromptType: The generated in-context example.
        """
        # Select the corresponding template
        if isinstance(self.template, str) or self.prompt_type == 'meta':
            tp = self.template
        else:
            # prompt type == origin
            tp = self.template[label]
        # tp = self._meta2str(tp, mode='ice')
        tp = self._encode_template(tp, ice=True)
        # Remove sep token
        if self.sep_token is not None:
            tp.replace(self.sep_token, '')
        # Remove ice_token
        if self.ice_token is not None:
            tp = tp.replace(self.ice_token, '')
        # Replace context token
        if isinstance(tp, str):
            # We want to use safe_substitute instead of str.format to avoid
            # KeyError while preserving the original string in curly brackets
            tp = safe_format(tp, **entry)
        else:
            tp = tp.format(**entry)
        return tp

    def generate_label_prompt_item(self,
                                   entry: Dict,
                                   ice: PromptType,
                                   label: Hashable,
                                   remain_sep: Optional[bool] = False) -> str:
        """Generate prompt based on :obj:`entry` data, :obj:`ice` in-context
        example, and the corresponding :obj:`label`.

        Args:

            entry (:obj:`Dict`): A piece of data containing the input field
                content.
            ice (PromptType): The generated in-context example.
            label (:obj:`Hashable`): The value of the output field.
            remain_sep (:obj:`bool`): If remain sep_token

        Returns:
            :obj:`str`: The generated prompt.
        """
        # Select the corresponding template
        if isinstance(self.template, str) or self.prompt_type == 'meta':
            template = self.template
        else:
            # template is a dict with a label -> prompt mapping
            template = self.template[label]
        template = self._encode_template(template, ice=False)
        # Remove sep token
        if not remain_sep and self.sep_token is not None:
            template = template.replace(self.sep_token, '')
        # Insert in-context examples
        if self.ice_token is not None:
            template = template.replace(self.ice_token, ice)
        # Replace context token
        if isinstance(template, str):
            # We want to use safe_substitute instead of str.format to avoid
            # KeyError while preserving the original string in curly brackets
            template = safe_format(template, **entry)
        else:
            template = template.format(**entry)
        return template

    def generate_item(
            self,
            entry: Dict,
            output_field: Optional[Hashable] = None,
            output_field_replace_token: Optional[str] = '',
            ice_field_replace_token: Optional[str] = '') -> PromptType:
        """Generate an item based on the provided :obj:`entry` data, as well as
        optional output field and ice field tokens.

        Warning:
            This method is only used in generation task, i.e. GenInferencer.

        Args:
            entry (:obj:`Dict`): A piece of data.
            output_field (:obj:`Hashable`, optional): Column name of output
                field. Defaults to :obj:`None`.
            output_field_replace_token (:obj:`str`, optional): Tokens used to
                replace output field. Defaults to ``''``.
            ice_field_replace_token (str, optional): Tokens used to replace
                the :obj:`ice_token`. Defaults to ``''``.

        Returns:
            PromptType: The generated item.
        """
        template = None
        if isinstance(self.template, str):
            template = self.template
        elif self.prompt_type == 'origin':
            # This if is only effective when you are using GenInferecner
            # with multi-label prompts.
            # Such a combination doesn't make sense at all :)
            # TODO: Check this, seems it is used in XXXRetriever as well
            template = self.template[list(self.template.keys())[0]]
            template = self._encode_template(template, ice=False)
        else:
            template = self._encode_template(self.template, ice=False)
        if self.ice_token is not None:
            template = template.replace(self.ice_token,
                                        ice_field_replace_token)
        # Remove sep token
        if self.sep_token is not None:
            template = template.replace(self.sep_token, '')
        if output_field is not None:
            entry = copy.deepcopy(entry)
            entry[output_field] = output_field_replace_token
        if isinstance(template, str):
            # We want to use safe_substitute instead of str.format to avoid
            # KeyError while preserving the original string in curly brackets
            template = safe_format(template, **entry)
        else:
            template = template.format(**entry)
        return template

    def _check_prompt_template(obj) -> 'PromptTemplate':
        if isinstance(obj, PromptTemplate):
            return obj
        else:
            raise TypeError(f'Expect a PromptTemplate object, but got {obj}')

    def __repr__(self):
        return (f'PromptTemplate({{\n\ttemplate: {self.template},\n\t'
                f'ice_token: {self.ice_token}\n}})')

    def _encode_template(self, prompt_template: Union[List[Union[str, Dict]],
                                                      str],
                         ice: bool) -> PromptType:
        """Encode the raw template given in the config into a str or a
        PromptList.

        Args:
            prompt_template (List[Dict]] or str): The raw template given in the
                config, used for generating the prompt. If it's a string, the
                result will be directly returned.
            ice (bool): If the template is used for generating in-context
                examples.

        Returns:
            PromptType: The encoded template.
        """
        if isinstance(prompt_template, str):
            return prompt_template

        prompt = PromptList()

        # TODO: Why can't we generate begin & end for ice template?
        # To fix this, first we need to allow specifying prompt_template
        # only
        if 'begin' in prompt_template and not ice:
            prompt.append(dict(section='begin', pos='begin'))
            if isinstance(prompt_template['begin'], list):
                prompt += prompt_template['begin']
            else:
                prompt.append(prompt_template['begin'])
            prompt.append(dict(section='begin', pos='end'))

        if ice:
            prompt.append(dict(section='ice', pos='begin'))
        else:
            prompt.append(dict(section='round', pos='begin'))
        prompt += prompt_template['round']
        if ice:
            prompt.append(dict(section='ice', pos='end'))
        else:
            prompt.append(dict(section='round', pos='end'))

        if 'end' in prompt_template and not ice:
            prompt.append(dict(section='end', pos='end'))
            if isinstance(prompt_template['end'], list):
                prompt += prompt_template['end']
            else:
                prompt.append(prompt_template['end'])
            prompt.append(dict(section='end', pos='end'))

        return prompt
