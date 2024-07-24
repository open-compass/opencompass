import asyncio
import json
import os
import re
import time
import uuid
import base64
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Union, Tuple, AsyncIterator, Any

import aiohttp
from dacite import from_dict

import requests
from requests.exceptions import HTTPError
from tqdm import tqdm

History = List[Union[Tuple[str, str], List[str]]]
Messages = List[Dict[str, Union[str, List[Dict]]]]


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


class XRequestConfig:
    """NOTE: The following behavior is inconsistent with the OpenAI API.
    Default values for OpenAI:
        temperature = 1.
        top_k = -1
        top_p = 1.
        repetition_penalty = 1.
    """
    max_tokens: Optional[int] = None  # None: max_model_len - num_tokens
    # None: use deploy_args
    temperature: Optional[float] = None
    top_p: Optional[float] = None

    n: int = 1
    seed: Optional[int] = None
    stop: Optional[List[str]] = None
    stream: bool = False

    best_of: Optional[int] = None
    presence_penalty: float = 0.
    frequency_penalty: float = 0.
    length_penalty: float = 1.

    # additional
    num_beams: int = 1
    # None: use deploy_args
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None


@dataclass
class Model:
    id: str  # model_type
    is_chat: Optional[bool] = None  # chat model or generation model
    is_multimodal: bool = False

    object: str = 'model'
    created: int = field(default_factory=lambda: int(time.time()))
    owned_by: str = 'swift'


@dataclass
class ModelList:
    data: List[Model]
    object: str = 'list'


@dataclass
class XRequestConfig:
    """NOTE: The following behavior is inconsistent with the OpenAI API.
    Default values for OpenAI:
        temperature = 1.
        top_k = -1
        top_p = 1.
        repetition_penalty = 1.
    """
    max_tokens: Optional[int] = None  # None: max_model_len - num_tokens
    # None: use deploy_args
    temperature: Optional[float] = None
    top_p: Optional[float] = None

    n: int = 1
    seed: Optional[int] = None
    stop: Optional[List[str]] = None
    stream: bool = False

    best_of: Optional[int] = None
    presence_penalty: float = 0.
    frequency_penalty: float = 0.
    length_penalty: float = 1.

    # additional
    num_beams: int = 1
    # None: use deploy_args
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None


@dataclass
class CompletionRequestMixin:
    model: str
    prompt: str
    images: List[str] = field(default_factory=list)


@dataclass
class ChatCompletionRequestMixin:
    model: str
    messages: List[Dict[str, Union[str, List[Dict]]]]
    tools: Optional[List[Dict[str, Union[str, Dict]]]] = None
    tool_choice: Optional[Union[str, Dict]] = 'auto'
    images: List[str] = field(default_factory=list)


@dataclass
class CompletionRequest(XRequestConfig, CompletionRequestMixin):
    pass


@dataclass
class ChatCompletionRequest(XRequestConfig, ChatCompletionRequestMixin):
    pass


@dataclass
class UsageInfo:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class Function:
    arguments: Optional[str] = None
    name: str = ''


@dataclass
class ChatCompletionMessageToolCall:
    id: str
    function: Function
    type: str = 'function'


@dataclass
class ChatMessage:
    role: Literal['system', 'user', 'assistant']
    content: str
    tool_calls: Optional[ChatCompletionMessageToolCall] = None


@dataclass
class ChatCompletionResponseChoice:
    index: int
    message: ChatMessage
    finish_reason: Literal['stop', 'length', None]  # None: for infer_backend='pt'


@dataclass
class CompletionResponseChoice:
    index: int
    text: str
    finish_reason: Literal['stop', 'length', None]  # None: for infer_backend='pt'


@dataclass
class ChatCompletionResponse:
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo
    id: str = field(default_factory=lambda: f'chatcmpl-{random_uuid()}')
    object: str = 'chat.completion'
    created: int = field(default_factory=lambda: int(time.time()))


@dataclass
class CompletionResponse:
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo
    id: str = field(default_factory=lambda: f'cmpl-{random_uuid()}')
    object: str = 'text_completion'
    created: int = field(default_factory=lambda: int(time.time()))


@dataclass
class DeltaMessage:
    role: Literal['system', 'user', 'assistant']
    content: str
    tool_calls: Optional[ChatCompletionMessageToolCall] = None


@dataclass
class ChatCompletionResponseStreamChoice:
    index: int
    delta: DeltaMessage
    finish_reason: Literal['stop', 'length', None]


@dataclass
class ChatCompletionStreamResponse:
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
    usage: UsageInfo
    id: str = field(default_factory=lambda: f'chatcmpl-{random_uuid()}')
    object: str = 'chat.completion.chunk'
    created: int = field(default_factory=lambda: int(time.time()))


@dataclass
class CompletionResponseStreamChoice:
    index: int
    text: str
    finish_reason: Literal['stop', 'length', None]


@dataclass
class CompletionStreamResponse:
    model: str
    choices: List[CompletionResponseStreamChoice]
    usage: UsageInfo
    id: str = field(default_factory=lambda: f'cmpl-{random_uuid()}')
    object: str = 'text_completion.chunk'
    created: int = field(default_factory=lambda: int(time.time()))


def get_model_list_client(host: str = '127.0.0.1', port: str = '8000', **kwargs) -> ModelList:
    url = kwargs.pop('url', None)
    if url is None:
        url = f'http://{host}:{port}/v1'
    url = url.rstrip('/')
    url = f'{url}/models'
    resp_obj = requests.get(url).json()
    return from_dict(ModelList, resp_obj)


def history_to_messages(history: Optional[History],
                        query: Optional[str] = None,
                        system: Optional[str] = None,
                        roles: Optional[List[List[str]]] = None) -> Messages:
    if history is None:
        history = []
    messages = []
    if not roles:
        roles = [['user', 'assistant']] * (len(history) + 1)
    assert len(roles) == len(history) + 1
    if system is not None:
        messages.append({'role': 'system', 'content': system})
    for role, h in zip(roles, history):
        assert isinstance(h, (list, tuple))
        messages.append({'role': role[0], 'content': h[0]})
        messages.append({'role': role[1], 'content': h[1]})
    if query is not None:
        messages.append({'role': roles[-1][0], 'content': query})
    return messages


def _to_base64(img_path: str) -> str:
    if not os.path.isfile(img_path):
        return img_path
    with open(img_path, 'rb') as f:
        img_base64: str = base64.b64encode(f.read()).decode('utf-8')
    return img_base64


def _encode_prompt(prompt: str) -> str:
    pattern = r'<(?:img|audio)>(.+?)</(?:img|audio)>'
    match_iter = re.finditer(pattern, prompt)
    new_prompt = ''
    idx = 0
    for m in match_iter:
        span = m.span(1)
        path = m.group(1)
        img_base64 = _to_base64(path)
        new_prompt += prompt[idx:span[0]] + img_base64
        idx = span[1]
    new_prompt += prompt[idx:]
    return new_prompt


def convert_to_base64(*,
                      messages: Optional[Messages] = None,
                      prompt: Optional[str] = None,
                      images: Optional[List[str]] = None) -> Dict[str, Any]:
    """local_path -> base64"""
    res = {}
    if messages is not None:
        res_messages = []
        for m in messages:
            m_new = deepcopy(m)
            m_new['content'] = _encode_prompt(m_new['content'])
            res_messages.append(m_new)
        res['messages'] = res_messages
    if prompt is not None:
        prompt = _encode_prompt(prompt)
        res['prompt'] = prompt
    if images is not None:
        res_images = []
        for image in images:
            res_images.append(_to_base64(image))
        res['images'] = res_images
    return res


def _parse_stream_data(data: bytes) -> Optional[str]:
    data = data.decode(encoding='utf-8')
    data = data.strip()
    if len(data) == 0:
        return
    assert data.startswith('data:'), f'data: {data}'
    return data[5:].strip()


class OpenAIClientUtil:

    @staticmethod
    def _pre_inference_client(model_type: str,
                              # query: str,
                              # history: Optional[History] = None,
                              # system: Optional[str] = None,
                              url: str,
                              messages: Optional[Messages] = None,
                              images: Optional[List[str]] = None,
                              *,
                              is_chat_request: Optional[bool] = True,
                              request_config: Optional[XRequestConfig] = None,
                              # host: str = '127.0.0.1',
                              # port: str = '8000',
                              **kwargs) -> Tuple[str, Dict[str, Any], bool]:
        if images is None:
            images = []

        is_multimodal = False  # TODO: to be supported

        # model_list = get_model_list_client(host, port, **kwargs)
        # for model in model_list.data:
        #     if model_type == model.id:
        #         _is_chat = model.is_chat
        #         is_multimodal = model.is_multimodal
        #         break
        # else:
        #     raise ValueError(f'model_type: {model_type}, model_list: {[model.id for model in model_list.data]}')

        data = {k: v for k, v in request_config.__dict__.items() if not k.startswith('__')}

        if is_chat_request:
            # messages = history_to_messages(history, query, system, kwargs.get('roles'))
            if is_multimodal:
                messages = convert_to_base64(messages=messages)['messages']
                images = convert_to_base64(images=images)['images']
            data['messages'] = messages
        else:
            # TODO: This is a temporary solution for non-chat models.
            input_prompts = []
            for msg in messages:
                input_prompts.append(msg['content'])
            query = '\n'.join(input_prompts)

            if is_multimodal:
                query = convert_to_base64(prompt=query)['prompt']
                images = convert_to_base64(images=images)['images']
            data['prompt'] = query

        data['model'] = model_type
        if len(images) > 0:
            data['images'] = images

        return url, data, is_chat_request

    @staticmethod
    async def _inference_client_async(
            model_type: str,
            # query: str,
            # history: Optional[History] = None,
            # system: Optional[str] = None,
            messages: Messages,
            url: str,
            images: Optional[List[str]] = None,
            *,
            is_chat_request: Optional[bool] = None,
            request_config: Optional[XRequestConfig] = None,
            # host: str = '127.0.0.1',
            # port: str = '8000',
            **kwargs
    ) -> Union[ChatCompletionResponse, CompletionResponse, AsyncIterator[ChatCompletionStreamResponse],
               AsyncIterator[CompletionStreamResponse]]:
        if request_config is None:
            request_config = XRequestConfig()

        url, data, is_chat_request = OpenAIClientUtil._pre_inference_client(
            model_type=model_type,
            url=url,
            messages=messages,
            images=images,
            is_chat_request=is_chat_request,
            request_config=request_config,
            # host=host,
            # port=port,
            **kwargs)

        if request_config.stream:
            if is_chat_request:
                ret_cls = ChatCompletionStreamResponse
            else:
                ret_cls = CompletionStreamResponse

            async def _gen_stream(
            ) -> Union[AsyncIterator[ChatCompletionStreamResponse], AsyncIterator[CompletionStreamResponse]]:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=data) as resp:
                        async for _data in resp.content:
                            _data = _parse_stream_data(_data)
                            if _data == '[DONE]':
                                break
                            if _data is not None:
                                resp_obj = json.loads(_data)
                                if resp_obj['object'] == 'error':
                                    raise HTTPError(resp_obj['message'])
                                yield from_dict(ret_cls, resp_obj)

            return _gen_stream()
        else:
            if is_chat_request:
                ret_cls = ChatCompletionResponse
            else:
                ret_cls = CompletionResponse
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as resp:
                    resp_obj = await resp.json()

                    if resp_obj['object'] == 'error':
                        raise HTTPError(resp_obj['message'])
                    return from_dict(ret_cls, resp_obj)

    @staticmethod
    async def _call_openai(model_type: str,
                           # query: str,
                           messages: Messages,
                           eval_url: str,
                           *,
                           is_chat_model: bool,
                           request_config: XRequestConfig,
                           prog_bar: tqdm,
                           ) -> Tuple[str, Optional[int]]:
        # idx: maintain the order
        resp = await OpenAIClientUtil._inference_client_async(
            model_type, messages, is_chat_request=is_chat_model, request_config=request_config, url=eval_url)
        if is_chat_model:
            response = resp.choices[0].message.content
        else:
            response = resp.choices[0].text
        prog_bar.update()
        return response

    @staticmethod
    async def call_openai_batched(model_type: str,
                                  # prompts: List[str],
                                  messages_batch: List[Messages],
                                  request_config: XRequestConfig,
                                  base_url: str,
                                  is_chat: bool = True,
                                  ) -> List[str]:
        use_tqdm = True if len(messages_batch) >= 20 else False
        prog_bar = tqdm(total=len(messages_batch), dynamic_ncols=True, disable=not use_tqdm)
        tasks = []
        for messages in messages_batch:
            tasks.append(
                OpenAIClientUtil._call_openai(
                    model_type,
                    # prompt,
                    messages,
                    base_url,
                    is_chat_model=is_chat,
                    request_config=request_config,
                    prog_bar=prog_bar))
        response_list: List[str] = await asyncio.gather(*tasks)
        prog_bar.close()
        return response_list
