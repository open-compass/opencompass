import os
import time
from threading import Lock
from typing import Dict, List, Optional, Union

from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList

from .openai_api import OpenAISDK

PromptType = Union[PromptList, str]
OPENAISDK_API_BASE = os.environ.get('OPENAI_BASE_URL',
                                    'https://api.openai.com/v1/')

O1_MODEL_LIST = ['o1', 'o3', 'o4']


@MODELS.register_module()
class OpenAISDKStreaming(OpenAISDK):
    """OpenAI SDK with streaming support for real-time token generation.

    This class extends OpenAISDK to support streaming responses where tokens
    are generated and returned as they become available, rather than waiting
    for the complete response.

    Args:
        stream (bool): Whether to enable streaming mode. Defaults to True.
        stream_chunk_size (int): Size of chunks for streaming. Defaults to 1.
        All other args are inherited from OpenAISDK.
    """

    def __init__(self,
                 path: str = 'gpt-3.5-turbo',
                 max_seq_len: int = 16384,
                 query_per_second: int = 1,
                 rpm_verbose: bool = False,
                 retry: int = 2,
                 key: str | List[str] = 'ENV',
                 org: str | List[str] | None = None,
                 meta_template: Dict | None = None,
                 openai_api_base: str | List[str] = OPENAISDK_API_BASE,
                 openai_proxy_url: Optional[str] = None,
                 mode: str = 'none',
                 logprobs: bool | None = False,
                 top_logprobs: int | None = None,
                 temperature: float | None = None,
                 tokenizer_path: str | None = None,
                 extra_body: Dict | None = None,
                 verbose: bool = False,
                 http_client_cfg: dict = {},
                 status_code_mappings: dict = {},
                 think_tag: str = '</think>',
                 stream: bool = True,
                 stream_chunk_size: int = 1):

        super().__init__(path=path,
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
                         think_tag=think_tag)

        self.stream = stream
        self.stream_chunk_size = stream_chunk_size

    def _create_fresh_client(self):
        """Create a fresh OpenAI client for each request to avoid
        concurrency issues."""
        import httpx
        from openai import OpenAI

        # Get current key (with key rotation)
        with Lock():
            if len(self.invalid_keys) == len(self.keys):
                raise RuntimeError('All keys have insufficient quota.')

            # find the next valid key
            while True:
                self.key_ctr += 1
                if self.key_ctr == len(self.keys):
                    self.key_ctr = 0

                if self.keys[self.key_ctr] not in self.invalid_keys:
                    break

            current_key = self.keys[self.key_ctr]

        # Create fresh client with current key
        http_client_cfg = {}
        if self.proxy_url:
            http_client_cfg['proxies'] = {
                'http://': self.proxy_url,
                'https://': self.proxy_url,
            }

        return OpenAI(
            base_url=self.openai_api_base,
            api_key=current_key,
            http_client=httpx.Client(
                **http_client_cfg,
                timeout=httpx.Timeout(3600.0)  # 1 hour timeout
            ) if http_client_cfg or True else None,
        )

    def _generate(
            self,
            input: PromptList | str,
            max_out_len: int,
            temperature: float,
            timeout: int = 3600,  # Set timeout to 1 hour
    ) -> str:
        """Generate results with streaming support.

        Args:
            input (PromptType): A string or PromptDict.
            max_out_len (int): The maximum length of the output.
            temperature (float): What sampling temperature to use.
            timeout (int, optional): Timeout in seconds for the API call.

        Returns:
            str: The generated string (complete response when streaming).
        """
        import threading

        from openai import APIStatusError, BadRequestError

        assert isinstance(input, (str, PromptList))

        messages, max_out_len = self._preprocess_messages(
            input, max_out_len, self.max_seq_len, self.mode,
            self.get_token_len)

        num_retries = 0
        while num_retries < self.retry:
            self.wait()

            if any(model in self.path for model in O1_MODEL_LIST):
                self.logger.warning(
                    f"'max_token' is unsupported for model {self.path}")
                self.logger.warning(
                    f'We use max_out_len: {max_out_len} for this query')
                query_data = dict(
                    model=self.path,
                    max_completion_tokens=max_out_len,
                    n=1,
                    messages=messages,
                    extra_body=self.extra_body,
                    stream=self.stream,  # Enable streaming
                )
            else:
                query_data = dict(
                    model=self.path,
                    max_tokens=max_out_len,
                    n=1,
                    temperature=self.temperature
                    if self.temperature is not None else temperature,
                    messages=messages,
                    extra_body=self.extra_body,
                    stream=self.stream,  # Enable streaming
                )

            try:
                if self.verbose:
                    thread_id = threading.get_ident()
                    self.logger.info(
                        f'[Thread {thread_id}] Start calling OpenAI API '
                        f'with streaming enabled')

                if self.stream:
                    # Create fresh client for each request to avoid
                    # concurrency issues
                    fresh_client = self._create_fresh_client()

                    # Handle streaming response with shorter timeout
                    response_stream = fresh_client.chat.completions.create(
                        **query_data, timeout=timeout)

                    result = self._handle_stream_response(
                        response_stream, thread_id if self.verbose else None)

                    # Clean up the client
                    if (hasattr(fresh_client, '_client')
                            and hasattr(fresh_client._client, 'close')):
                        fresh_client._client.close()

                    return result
                else:
                    # Fallback to non-streaming (use parent method)
                    return super()._generate(input, max_out_len, temperature,
                                             timeout)

            except (BadRequestError, APIStatusError) as e:
                status_code = e.status_code
                if (status_code is not None
                        and status_code in self.status_code_mappings):
                    error_message = self.status_code_mappings[status_code]
                    self.logger.error(
                        f'error occurs at {self.openai_api_base}')
                    self.logger.info(f'Status Code: {status_code}, \n'
                                     f'Original Error Message: {e}, \n'
                                     f'Return Message: {error_message} ')
                    return error_message
                else:
                    self.logger.error(
                        f'error occurs at {self.openai_api_base}')
                    self.logger.error(e)
            except Exception as e:
                thread_id = threading.get_ident()
                self.logger.error(f'[Thread {thread_id}] error occurs at '
                                  f'{self.openai_api_base}: {e}')
                import traceback
                self.logger.error(f'[Thread {thread_id}] Traceback: '
                                  f'{traceback.format_exc()}')
            num_retries += 1

        raise RuntimeError('Calling OpenAI API failed after retrying for '
                           f'{self.retry} times. Check the logs for details.')

    def _handle_stream_response(self, response_stream, thread_id=None) -> str:
        """Handle streaming response and collect all chunks into final
        response.

        Args:
            response_stream: The streaming response from OpenAI API
            thread_id: Optional thread ID for logging

        Returns:
            str: Complete generated text from all chunks
        """
        completion_chunks = []
        reasoning_content = ''
        chunk_count = 0
        start_time = time.time()

        def log_with_thread(message, level='info'):
            if thread_id:
                message = f'[Thread {thread_id}] {message}'
            getattr(self.logger, level)(message)

        try:
            if self.verbose:
                log_with_thread('Starting to process streaming response')

            for chunk in response_stream:
                chunk_count += 1
                current_time = time.time()

                # Add timeout check for stuck streams
                # 1 hour timeout for streaming
                if current_time - start_time > 3600:
                    log_with_thread(
                        f'Streaming timeout after '
                        f'{current_time - start_time:.1f}s, '
                        f'chunks processed: {chunk_count}', 'warning')
                    break

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # Handle reasoning content if present
                if (hasattr(delta, 'reasoning_content')
                        and delta.reasoning_content):
                    reasoning_content += delta.reasoning_content

                # Handle regular content
                if delta.content:
                    completion_chunks.append(delta.content)
                    if self.verbose:
                        # Print streaming output in real-time with complete
                        # content
                        print(delta.content, end='', flush=True)

                # Check if streaming is finished
                if chunk.choices[0].finish_reason is not None:
                    if self.verbose:
                        print()  # Add newline after streaming complete
                        elapsed = current_time - start_time
                        log_with_thread(
                            f'Streaming finished with reason: '
                            f'{chunk.choices[0].finish_reason}, '
                            f'chunks: {chunk_count}, elapsed: {elapsed:.1f}s')
                    break

        except Exception as e:
            elapsed = time.time() - start_time
            log_with_thread(
                f'Error during streaming after {elapsed:.1f}s, '
                f'chunks: {chunk_count}: {e}', 'error')
            import traceback
            log_with_thread(
                f'Streaming error traceback: {traceback.format_exc()}',
                'error')

            # Return whatever we've collected so far
            if not completion_chunks and not reasoning_content:
                raise  # Re-raise if we got nothing
        finally:
            # Ensure we close the stream
            try:
                if hasattr(response_stream, 'close'):
                    response_stream.close()
            except Exception:
                pass

        # Combine reasoning content and regular content
        complete_content = ''.join(completion_chunks)

        if self.verbose:
            log_with_thread(f'Stream processing complete. Content length: '
                            f'{len(complete_content)}, '
                            f'Reasoning length: {len(reasoning_content)}')

        if reasoning_content:
            if self.verbose:
                log_with_thread(
                    'Streaming with reasoning content detected.\n'
                    f'Reasoning Content length: {len(reasoning_content)} \n'
                    f'Tags: {self.think_tag} \n'
                    f'Content length: {len(complete_content)}', )
            if complete_content:
                return reasoning_content + self.think_tag + complete_content
            else:
                return reasoning_content
        else:
            return complete_content

    def estimate_token_count(self, text: str) -> int:
        """Estimate token count for given text.

        Args:
            text (str): Text to count tokens for

        Returns:
            int: Estimated token count
        """
        return self.get_token_len(text)
