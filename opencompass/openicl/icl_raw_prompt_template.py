from typing import Dict, List, Optional

from opencompass.registry import ICL_PROMPT_TEMPLATES
from opencompass.utils.prompt import safe_format

MessageType = List[Dict[str, str]]  # [{'role': 'user', 'content': '...'}, ...]


@ICL_PROMPT_TEMPLATES.register_module()
class RawPromptTemplate:
    """直接透传 messages 格式的 PromptTemplate。

    绕过 PromptList 和 APITemplateParser 的多层转换，
    直接使用 OpenAI 兼容的 messages 格式。

    Args:
        messages (List[Dict]): OpenAI 格式的 messages 列表
            [{'role': 'system', 'content': '...'},
             {'role': 'user', 'content': '{input}'},
             ...]
        format_variables (bool): 是否对 content 中的变量进行替换。默认 True

    """

    def __init__(
        self,
        messages: List[Dict[str, str]],
        format_variables: bool = True,
    ) -> None:
        self._validate_messages(messages)
        self.messages = messages
        self.format_variables = format_variables
        self.prompt_type = 'raw_messages'

        # 兼容性：设置 ice_token 为 None
        self.ice_token = None
        self.sep_token = None

    def _validate_messages(self, messages: List[Dict]) -> None:
        """验证 messages 格式"""
        if not isinstance(messages, list):
            raise TypeError(f'messages must be a list, got {type(messages)}')

        valid_roles = {'system', 'user', 'assistant'}
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise TypeError(
                    f'messages[{i}] must be a dict, got {type(msg)}')
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
        """生成最终的 messages

        Args:
            entry: 数据条目，包含 {input}, {output} 等字段
            output_field: 输出字段名（兼容现有接口，暂不使用）
            output_field_replace_token: 输出字段替换 token（兼容现有接口）
            ice_field_replace_token: ice 替换 token（兼容现有接口）

        Returns:
            List[Dict]: OpenAI 格式的 messages
        """
        import copy
        result = copy.deepcopy(self.messages)

        if self.format_variables:
            for msg in result:
                if 'content' in msg and isinstance(msg['content'], str):
                    msg['content'] = safe_format(msg['content'], **entry)
        return result

    def generate_ice_item(self,
                          entry: Dict,
                          label=None) -> List[Dict[str, str]]:
        """生成 in-context example（兼容接口）

        对于 RawPromptTemplate，通常不用于生成 ICE，
        但为了兼容性保留此方法。
        """
        return self.generate_item(entry)

    def generate_label_prompt_item(
        self,
        entry: Dict,
        ice: str = '',
        label=None,
        remain_sep: bool = False,
    ) -> List[Dict[str, str]]:
        """生成带标签的 prompt（兼容接口）"""
        return self.generate_item(entry)

    def __repr__(self):
        return f'RawPromptTemplate(messages={self.messages})'
