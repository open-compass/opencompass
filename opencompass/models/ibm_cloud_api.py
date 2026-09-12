import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

import requests

from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]


@MODELS.register_module()
class IBMGranite(BaseAPIModel):
    """Model wrapper for IBM Granite models served on watsonx.ai.

    Args:
        path (str): Watsonx foundation model identifier, e.g.
            ``'ibm/granite-8b-code-instruct'``.
        api_key (str): IBM Cloud API key. When set to ``'ENV'`` the key is
            fetched from ``$IBM_CLOUD_API_KEY``.
        project_id (str, optional): Watsonx project identifier. When set to
            ``'ENV'`` the id is fetched from ``$IBM_CLOUD_PROJECT_ID``.
        space_id (str, optional): Watsonx space identifier. When set to
            ``'ENV'`` the id is fetched from ``$IBM_CLOUD_SPACE_ID``. Either
            ``project_id`` or ``space_id`` must be provided.
        region (str): Watsonx region slug. Defaults to ``'us-south'``.
        version (str): API version query string appended to each request.
            Defaults to ``'2024-05-29'`` which is the latest stable watsonx
            text generation API version at the time of writing.
        query_per_second (int): Rate limit for outgoing requests.
        meta_template (Dict, optional): Meta prompt template configuration.
        retry (int): Number of retries for failed requests. Defaults to 2.
        parameters (Dict, optional): Default generation parameters passed to
            the API.
        iam_url (str): URL used to exchange API key for IAM access token.
        timeout (float): Request timeout in seconds. Defaults to 30 seconds.
        max_workers (int, optional): Max workers for the thread pool used to
            dispatch concurrent requests. Defaults to ``None`` (use
            ``ThreadPoolExecutor`` default).
    """

    def __init__(
        self,
        path: str,
        api_key: str,
        project_id: Optional[str] = None,
        space_id: Optional[str] = None,
        *,
        region: str = 'us-south',
        version: str = '2024-05-29',
        query_per_second: int = 1,
        rpm_verbose: bool = False,
        max_seq_len: int = 4096,
        meta_template: Optional[Dict] = None,
        retry: int = 2,
        parameters: Optional[Dict] = None,
        iam_url: str = 'https://iam.cloud.ibm.com/identity/token',
        timeout: float = 120.0,
        max_workers: Optional[int] = None,
        generation_kwargs: Optional[Dict] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            path=path,
            max_seq_len=max_seq_len,
            query_per_second=query_per_second,
            rpm_verbose=rpm_verbose,
            meta_template=meta_template,
            retry=retry,
            generation_kwargs=generation_kwargs or {},
            verbose=verbose,
        )

        if api_key == 'ENV':
            api_key = os.getenv('IBM_CLOUD_API_KEY')
            if not api_key:
                raise ValueError('IBM Cloud API key is not set.')
        self.api_key = api_key

        if project_id == 'ENV':
            project_id = os.getenv('IBM_CLOUD_PROJECT_ID')
        if space_id == 'ENV':
            space_id = os.getenv('IBM_CLOUD_SPACE_ID')
        if not project_id and not space_id:
            raise ValueError('Either project_id or space_id must be provided.')

        self.project_id = project_id
        self.space_id = space_id
        self.parameters = parameters.copy() if parameters else {}
        self.iam_url = iam_url
        self.timeout = timeout
        self.url = f'https://{region}.ml.cloud.ibm.com/ml/v1/text/generation'
        if not version:
            raise ValueError(
                'IBM Granite text generation endpoint requires a version ' \
                'query parameter.')
        self.version = version
        self._request_params = {'version': version}
        self.max_workers = max_workers

        self._token: Optional[str] = None
        self._token_expiry: float = 0.0
        self._token_lock = threading.Lock()

    def generate(
        self,
        inputs: List[PromptType],
        max_out_len: int = 2046,
        **kwargs,
    ) -> List[str]:
        if kwargs:
            request_overrides = [dict(kwargs) for _ in inputs]
        else:
            request_overrides = [{} for _ in inputs]
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(
                executor.map(
                    self._generate,
                    inputs,
                    [max_out_len] * len(inputs),
                    request_overrides,
                ))
        self.flush()
        return results

    def _generate(
        self, input: PromptType, max_out_len: int, request_overrides: Dict,
    ) -> str:
        assert isinstance(input, (str, PromptList))
        prompt = self._convert_prompt(input)
        payload = {
            'model_id': self.path,
            'input': prompt,
            'parameters': self._build_parameters(
                max_out_len, request_overrides),
        }
        if self.project_id:
            payload['project_id'] = self.project_id
        if self.space_id:
            payload['space_id'] = self.space_id

        num_retries = 0
        last_error = None
        while num_retries <= self.retry:
            try:
                token = self._get_token()
            except RuntimeError as err:
                last_error = err
                break

            headers = {
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json',
                'Accept': 'application/json',
            }

            self.acquire()
            try:
                response = requests.post(
                    self.url,
                    headers=headers,
                    json=payload,
                    params=self._request_params,
                    timeout=self.timeout,
                )
            except requests.RequestException as err:
                last_error = err
                self.logger.error(f'IBM Granite request error: {err}')
                self.release()
                time.sleep(2 ** num_retries)
                num_retries += 1
                continue

            try:
                data = response.json()
            except ValueError:
                data = None

            self.release()

            if response.status_code == 401:
                self.logger.warning('IBM Granite token rejected, refreshing...')
                self._invalidate_token()
                num_retries += 1
                continue
            if response.status_code in (429, 503):
                self.logger.warning('IBM Granite rate limited, retrying...')
                time.sleep(2 ** num_retries)
                num_retries += 1
                continue
            if response.status_code >= 400:
                error_msg = data.get('errors') if isinstance(data, dict) else None
                self.logger.error(
                    'IBM Granite request failed with status %s: %s',
                    response.status_code,
                    error_msg or response.text,
                )
                last_error = RuntimeError(response.text)
                num_retries += 1
                time.sleep(2 ** num_retries)
                continue

            if not isinstance(data, dict):
                last_error = RuntimeError('Invalid response from IBM Granite API')
                num_retries += 1
                time.sleep(2 ** num_retries)
                continue

            generated = self._extract_generated_text(data)
            if generated is not None:
                return generated

            last_error = RuntimeError(
                f'Unexpected response structure from IBM Granite API: {data}')
            num_retries += 1
            time.sleep(2 ** num_retries)

        if last_error:
            raise RuntimeError(last_error)
        raise RuntimeError('Failed to generate response from IBM Granite API')

    def _convert_prompt(self, prompt: PromptType) -> str:
        if isinstance(prompt, str):
            return prompt
        segments: List[str] = []
        for item in prompt:
            content = item.get('prompt', '')
            if not content:
                continue
            role = item.get('role') or item.get('api_role')
            if role:
                segments.append(f"{role.lower()}: {content}")
            else:
                segments.append(content)
        return '\n'.join(segments)

    def _build_parameters(
        self, max_out_len: int, request_overrides: Optional[Dict] = None,
    ) -> Dict:
        params = dict(self.parameters)
        params.update(self.generation_kwargs)
        if request_overrides:
            params.update({
                key: value for key, value in request_overrides.items()
                if value is not None
            })
        if max_out_len is not None and 'max_new_tokens' not in params:
            params['max_new_tokens'] = max_out_len
        params.setdefault('decoding_method', 'greedy')
        if params.get('decoding_method') == 'greedy':
            params.pop('temperature', None)
            params.pop('top_p', None)
        return params

    def _get_token(self) -> str:
        with self._token_lock:
            current_time = time.time()
            if self._token and current_time < self._token_expiry - 60:
                return self._token

            headers = {'Content-Type': 'application/x-www-form-urlencoded'}
            data = {
                'grant_type': 'urn:ibm:params:oauth:grant-type:apikey',
                'apikey': self.api_key,
            }
            try:
                response = requests.post(
                    self.iam_url,
                    headers=headers,
                    data=data,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                payload = response.json()
            except requests.RequestException as err:
                self.logger.error(f'Failed to obtain IBM IAM token: {err}')
                raise RuntimeError(err) from err

            access_token = payload.get('access_token')
            expires_in = payload.get('expires_in', 3600)
            if not access_token:
                raise RuntimeError('IBM IAM response missing access_token')

            self._token = access_token
            self._token_expiry = current_time + float(expires_in)
            return self._token

    def _invalidate_token(self) -> None:
        with self._token_lock:
            self._token = None
            self._token_expiry = 0.0

    def _extract_generated_text(self, response: Dict) -> Optional[str]:
        results = response.get('results')
        if isinstance(results, list) and results:
            generated = results[0].get('generated_text')
            if generated is not None:
                return generated.strip()
        outputs = response.get('outputs')
        if isinstance(outputs, list) and outputs:
            generated = outputs[0].get('text') or outputs[0].get('generated_text')
            if generated is not None:
                return generated.strip()
        return None

    def get_ppl(self, inputs: List[PromptType], mask_length=None) -> List[float]:
        raise NotImplementedError('IBM Granite API does not support PPL mode')