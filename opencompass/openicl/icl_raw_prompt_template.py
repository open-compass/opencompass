import copy
from typing import Dict, List, Optional, Union

from opencompass.registry import ICL_PROMPT_TEMPLATES
from opencompass.utils.prompt import safe_format

MessageType = List[Dict[str, str]]  # [{'role': 'user', 'content': '...'}, ...]
# 支持字符串元素作为 ICE 占位符
MessageElementType = Union[Dict[str, str], str]


@ICL_PROMPT_TEMPLATES.register_module()
class RawPromptTemplate:
    """直接透传 messages 格式的 PromptTemplate，支持 Few-shot。

    绕过 PromptList 和 APITemplateParser 的多层转换，
    直接使用 OpenAI 兼容的 messages 格式。

    Args:
        messages (List[Union[Dict, str]]): OpenAI 格式的 messages 列表
            支持两种元素类型:
            - Dict: 标准的 message，如 {'role': 'user', 'content': '...'}
            - str: ICE 占位符字符串，如 '</E>'
            示例:
            [{'role': 'system', 'content': '...'},
             '</E>',  # ICE 插入位置
             {'role': 'user', 'content': '{input}'},
             {'role': 'assistant', 'content': ''}]
        format_variables (bool): 是否对 content 中的变量进行替换。默认 True
        ice_token (str): ICE 占位符标识，默认 '</E>'
            messages 中的字符串元素如果等于 ice_token，将被替换为 ICE 内容

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
        """验证 messages 格式"""
        if not isinstance(messages, list):
            raise TypeError(f'messages must be a list, got {type(messages)}')

        valid_roles = {'system', 'user', 'assistant'}
        for i, msg in enumerate(messages):
            if isinstance(msg, str):
                # 允许字符串元素作为 ICE 占位符
                continue
            if not isinstance(msg, dict):
                raise TypeError(
                    f'messages[{i}] must be a dict or str, got {type(msg)}')
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
            ice_field_replace_token: ice 替换 token
                - 如果是 str: 将插入到 ice_token 位置
                - 如果是 List[Dict]: 将展开并插入到 ice_token 位置

        Returns:
            List[Dict]: OpenAI 格式的 messages
        """
        result = []

        for item in self.messages:
            if isinstance(item, str):
                # 字符串元素：检查是否是 ICE 占位符
                if item == self.ice_token and ice_field_replace_token:
                    if isinstance(ice_field_replace_token, list):
                        # ICE 是 messages 列表，展开插入
                        result.extend(ice_field_replace_token)
                    elif isinstance(ice_field_replace_token, str):
                        # ICE 是字符串，忽略（或可以选择插入为 user message）
                        # 这里选择跳过字符串 ICE，因为 messages 格式需要 dict
                        pass
                else:
                    # 非 ICE 占位符的字符串，跳过
                    pass
            else:
                # Dict 元素：深拷贝并格式化
                msg = copy.deepcopy(item)
                if self.format_variables:
                    if 'content' in msg and isinstance(msg['content'], str):
                        msg['content'] = safe_format(msg['content'], **entry)
                result.append(msg)

        return result

    def generate_ice_item(self,
                          entry: Dict,
                          label=None) -> List[Dict[str, str]]:
        """生成 in-context example

        只处理 messages 中的 Dict 元素，忽略字符串占位符。

        Args:
            entry: 数据条目，包含 {input}, {output} 等字段
            label: 标签（兼容现有接口）

        Returns:
            List[Dict]: OpenAI 格式的 messages（不包含 ICE 占位符）
        """
        result = []
        for item in self.messages:
            if isinstance(item, dict):
                msg = copy.deepcopy(item)
                if self.format_variables:
                    if 'content' in msg and isinstance(msg['content'], str):
                        msg['content'] = safe_format(msg['content'], **entry)
                result.append(msg)
            # 字符串元素（ICE 占位符）在生成 ICE 时被忽略
        return result

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
        return f'RawPromptTemplate(messages={self.messages}, ice_token={self.ice_token})'  # noqa
