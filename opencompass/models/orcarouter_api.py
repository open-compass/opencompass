"""OrcaRouter AI gateway backend for OpenCompass.

OrcaRouter is an OpenAI-compatible AI gateway built for both models and
agents. Like OpenRouter, it exposes a provider/model namespace across many
models, but it also combines adaptive routing, automatic failover,
zero-markup inference, observability, guardrails, and agent-tool governance
behind the same endpoint. It runs gateway-level, zero-trust security for AI
agents on the same endpoint, screening every prompt/response and governing
every tool call on a default-deny basis, with no application code changes.

This wrapper makes ``orcarouter/*`` and third-party models available through
the OrcaRouter gateway. Target models use the OrcaRouter namespace convention,
e.g. ``orcarouter/fusion``, ``orcarouter/free``, or any model id exposed by
``GET https://api.orcarouter.ai/v1/models``.

See https://www.orcarouter.ai for the full model list and API reference.
"""

import os
from typing import Dict, List, Optional, Union

from opencompass.registry import MODELS

from .openai_api import OpenAISDK
from .openai_streaming import OpenAISDKStreaming

#: Default OrcaRouter gateway base URL (OpenAI-compatible ``/v1`` endpoint).
ORCAROUTER_API_BASE = os.environ.get('ORCAROUTER_API_BASE',
                                     'https://api.orcarouter.ai/v1/')


@MODELS.register_module()
class OrcaRouterAPI(OpenAISDK):
    """Model wrapper around the OrcaRouter AI gateway.

    ``OrcaRouterAPI`` mirrors the existing ``OpenAISDK`` integration and
    points it at the OrcaRouter gateway, so all ``OpenAISDK`` arguments
    (``mode``, ``meta_template``, ``extra_body``, ``openai_extra_kwargs``,
    ...) apply unchanged.

    Args:
        path (str): Model id exposed by the OrcaRouter gateway, e.g.
            ``orcarouter/fusion``. Defaults to ``orcarouter/free``.
        key (str): OrcaRouter API key. When ``'ENV'`` (default), the key is
            read from the ``ORCAROUTER_API_KEY`` environment variable.
        openai_api_base (str | List[str]): The gateway base URL. Defaults to
            ``ORCAROUTER_API_BASE``.
        All other args are inherited from ``OpenAISDK``.
    """

    def __init__(
        self,
        path: str = 'orcarouter/free',
        max_seq_len: int = 16384,
        query_per_second: int = 1,
        rpm_verbose: bool = False,
        retry: int = 2,
        key: str = 'ENV',
        org: Union[str, List[str], None] = None,
        meta_template: Optional[Dict] = None,
        openai_api_base: Union[str, List[str]] = ORCAROUTER_API_BASE,
        openai_proxy_url: Optional[str] = None,
        mode: str = 'none',
        logprobs: Optional[bool] = False,
        top_logprobs: Optional[int] = None,
        temperature: Optional[float] = None,
        tokenizer_path: Optional[str] = None,
        extra_body: Optional[Dict] = None,
        verbose: bool = False,
        http_client_cfg: dict = {},
        status_code_mappings: dict = {},
        think_tag: str = '</think>',
        max_workers: Optional[int] = None,
        openai_extra_kwargs: Optional[Dict] = None,
        timeout: int = 3600,
        image_format: Optional[str] = None,
        image_min_edge: Optional[int] = None,
    ):
        if key == 'ENV':
            if 'ORCAROUTER_API_KEY' not in os.environ:
                raise ValueError('OrcaRouter API key is not set.')
            key = os.environ.get('ORCAROUTER_API_KEY')
        super().__init__(
            path=path,
            max_seq_len=max_seq_len,
            query_per_second=query_per_second,
            rpm_verbose=rpm_verbose,
            retry=retry,
            key=key,
            org=org,
            meta_template=meta_template,
            openai_api_base=openai_api_base,
            openai_proxy_url=openai_proxy_url,
            mode=mode,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            temperature=temperature,
            tokenizer_path=tokenizer_path,
            extra_body=extra_body,
            verbose=verbose,
            http_client_cfg=http_client_cfg,
            status_code_mappings=status_code_mappings,
            think_tag=think_tag,
            max_workers=max_workers,
            openai_extra_kwargs=openai_extra_kwargs,
            timeout=timeout,
            image_format=image_format,
            image_min_edge=image_min_edge,
        )


@MODELS.register_module()
class OrcaRouterAPIStreaming(OpenAISDKStreaming):
    """Streaming variant of :class:`OrcaRouterAPI`.

    Wraps :class:`OpenAISDKStreaming` against the OrcaRouter gateway. All
    arguments are inherited from :class:`OpenAISDKStreaming`, except
    ``path`` (defaults to ``orcarouter/free``), ``key`` (defaults to
    ``ORCAROUTER_API_KEY`` env var), and ``openai_api_base`` (defaults to
    ``ORCAROUTER_API_BASE``).
    """

    def __init__(
        self,
        path: str = 'orcarouter/free',
        max_seq_len: int = 16384,
        query_per_second: int = 1,
        rpm_verbose: bool = False,
        retry: int = 2,
        key: str = 'ENV',
        org: Union[str, List[str], None] = None,
        meta_template: Optional[Dict] = None,
        openai_api_base: Union[str, List[str]] = ORCAROUTER_API_BASE,
        openai_proxy_url: Optional[str] = None,
        mode: str = 'none',
        logprobs: Optional[bool] = False,
        top_logprobs: Optional[int] = None,
        temperature: Optional[float] = None,
        tokenizer_path: Optional[str] = None,
        extra_body: Optional[Dict] = None,
        verbose: bool = False,
        http_client_cfg: dict = {},
        status_code_mappings: dict = {},
        think_tag: str = '</think>',
        openai_extra_kwargs: Optional[Dict] = None,
        stream: bool = True,
        stream_chunk_size: int = 1,
        timeout: int = 3600,
        finish_reason_confirm: bool = True,
        max_workers: Optional[int] = None,
        image_format: Optional[str] = None,
        image_min_edge: Optional[int] = None,
    ):
        if key == 'ENV':
            if 'ORCAROUTER_API_KEY' not in os.environ:
                raise ValueError('OrcaRouter API key is not set.')
            key = os.environ.get('ORCAROUTER_API_KEY')
        super().__init__(
            path=path,
            max_seq_len=max_seq_len,
            query_per_second=query_per_second,
            rpm_verbose=rpm_verbose,
            retry=retry,
            key=key,
            org=org,
            meta_template=meta_template,
            openai_api_base=openai_api_base,
            openai_proxy_url=openai_proxy_url,
            mode=mode,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            temperature=temperature,
            tokenizer_path=tokenizer_path,
            extra_body=extra_body,
            verbose=verbose,
            http_client_cfg=http_client_cfg,
            status_code_mappings=status_code_mappings,
            think_tag=think_tag,
            openai_extra_kwargs=openai_extra_kwargs,
            stream=stream,
            stream_chunk_size=stream_chunk_size,
            timeout=timeout,
            finish_reason_confirm=finish_reason_confirm,
            max_workers=max_workers,
            image_format=image_format,
            image_min_edge=image_min_edge,
        )
