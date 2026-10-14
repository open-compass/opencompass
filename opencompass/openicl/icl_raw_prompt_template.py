import copy
from typing import Dict, List, Optional, Union

from opencompass.registry import ICL_PROMPT_TEMPLATES
from opencompass.utils.prompt import safe_format

MessageType = List[Dict[str, str]]  # [{'role': 'user', 'content': '...'}, ...]
# String elements are allowed as ICE placeholders
MessageElementType = Union[Dict[str, str], str]


@ICL_PROMPT_TEMPLATES.register_module()
class RawPromptTemplate:
    """PromptTemplate that directly passes through messages format,
    supporting Few-shot.

    Bypasses the multi-layer conversion of PromptList and
    APITemplateParser, directly using OpenAI-compatible messages format.

    Args:
        messages (List[Union[Dict, str]]): OpenAI-format messages list.
            Supports three element types:
            - Dict: Standard message, e.g.
              {'role': 'user', 'content': '...'}
            - Dict with 'expand_column': Dynamic expansion placeholder,
              reads a List[Dict] from a dataset column and expands it
              into multiple messages. For example,
              {'expand_column': 'history'} will expand
              entry['history'] containing
              [{'role': 'user', 'content': '...'}, ...] and insert them.
            - str: ICE placeholder string, e.g. '</E>'
            Example:
            [{'role': 'system', 'content': '...'},
             {'expand_column': 'history'},  # dynamically expand
                                            # multi-turn dialogue history
             {'role': 'user', 'content': '{question}'},
             {'role': 'assistant', 'content': ''}]
        format_variables (bool): Whether to substitute variables in content.
            Defaults to True.
        ice_token (str): ICE placeholder identifier. Defaults to '</E>'.
            String elements in messages that equal ice_token will be
            replaced with ICE content.

    """

    def __init__(
        self,
        messages: List[MessageElementType],
        format_variables: bool = True,
        ice_token: str = '</E>',
    ) -> None:
        self._validate_messages(messages)
        self.messages = messages
        self.format_variables = format_variables
        self.ice_token = ice_token
        self.sep_token = None
        self.prompt_type = 'raw_messages'

    def _validate_messages(self, messages: List) -> None:
        """Validate messages format."""
        if not isinstance(messages, list):
            raise TypeError(f'messages must be a list, got {type(messages)}')

        valid_roles = {'system', 'user', 'assistant'}
        for i, msg in enumerate(messages):
            if isinstance(msg, str):
                # Allow string elements as ICE placeholders
                continue
            if not isinstance(msg, dict):
                raise TypeError(
                    f'messages[{i}] must be a dict or str, got {type(msg)}')
            # Allow expand_column elements, no role/content required
            if 'expand_column' in msg:
                continue
            if 'role' not in msg:
                raise ValueError(f"messages[{i}] missing 'role' key")
            if 'content' not in msg:
                raise ValueError(f"messages[{i}] missing 'content' key")
            if msg['role'] not in valid_roles:
                raise ValueError(
                    f"messages[{i}] has invalid role: {msg['role']}")

    def generate_item(
        self,
        entry: Dict,
        output_field: Optional[str] = None,
        output_field_replace_token: Optional[str] = '',
        ice_field_replace_token: Optional[str] = '',
    ) -> List[Dict[str, str]]:
        """Generate the final messages list.

        Args:
            entry: A data entry containing fields like {input}, {output}, etc.
            output_field: Output field name (for interface compatibility,
                currently unused).
            output_field_replace_token: Output field replacement token
                (for interface compatibility).
            ice_field_replace_token: ICE replacement token.
                - If str: will be inserted at the ice_token position.
                - If List[Dict]: will be expanded and inserted at the
                  ice_token position.

        Returns:
            List[Dict]: OpenAI-format messages.
        """
        result = []

        for item in self.messages:
            if isinstance(item, str):
                # String element: check if it is an ICE placeholder
                if item == self.ice_token and ice_field_replace_token:
                    if isinstance(ice_field_replace_token, list):
                        # ICE is a messages list, expand and insert
                        result.extend(ice_field_replace_token)
                    elif isinstance(ice_field_replace_token, str):
                        # ICE is a string, skip since messages format
                        # requires dicts
                        pass
                else:
                    # Non-ICE placeholder strings are skipped
                    pass
            elif isinstance(item, dict) and 'expand_column' in item:
                # Dynamic expansion: read messages list from dataset
                # column and insert
                col = item['expand_column']
                if col in entry and isinstance(entry[col], list):
                    result.extend(copy.deepcopy(entry[col]))
            else:
                # Dict element: deep copy and format variables
                msg = copy.deepcopy(item)
                if self.format_variables:
                    if 'content' in msg and isinstance(msg['content'], str):
                        msg['content'] = safe_format(msg['content'], **entry)
                result.append(msg)

        return result

    def generate_ice_item(self,
                          entry: Dict,
                          label=None) -> List[Dict[str, str]]:
        """Generate in-context example.

        Only processes Dict elements in messages, ignoring string
        placeholders.

        Args:
            entry: A data entry containing fields like {input}, {output},
                etc.
            label: Label (for interface compatibility).

        Returns:
            List[Dict]: OpenAI-format messages (without ICE placeholders).
        """
        result = []
        for item in self.messages:
            if isinstance(item, dict) and 'expand_column' in item:
                # Dynamic expansion: read messages list from dataset
                # column and insert
                col = item['expand_column']
                if col in entry and isinstance(entry[col], list):
                    result.extend(copy.deepcopy(entry[col]))
            elif isinstance(item, dict):
                msg = copy.deepcopy(item)
                if self.format_variables:
                    if 'content' in msg and isinstance(msg['content'], str):
                        msg['content'] = safe_format(msg['content'], **entry)
                result.append(msg)
            # String elements (ICE placeholders) are ignored when
            # generating ICE
        return result

    def generate_label_prompt_item(
        self,
        entry: Dict,
        ice: str = '',
        label=None,
        remain_sep: bool = False,
    ) -> List[Dict[str, str]]:
        """Generate labeled prompt (interface compatibility)."""
        return self.generate_item(entry)

    def __repr__(self):
        return f'RawPromptTemplate(messages={self.messages}, ice_token={self.ice_token})'  # noqa
