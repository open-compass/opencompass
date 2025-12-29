# flake8: noqa

import hashlib
import hmac
import json
import urllib.parse


class Authorization:

    def __init__(self):
        self.SIGNED_HEADERS = 'content-type'  # 默认签名头
        pass

    # 编码函数
    def normalize(self, string, encodingSlash=True):
        quoted_string = urllib.parse.quote(
            string, safe='~()*!\'/' if encodingSlash else '~()*!\'')
        return quoted_string

    # 生成Canonical URI
    def generate_canonical_uri(self, url):
        parsed_url = urllib.parse.urlparse(url)
        return '/'.join(
            self.normalize(segment, False)
            for segment in parsed_url.path.split('/'))

    # 生成Canonical Headers
    def generate_canonical_headers(self, headers):
        signed_headers_list = self.SIGNED_HEADERS.split(';')
        signed_headers_set = set(signed_headers_list)
        sorted_headers = sorted(
            (k.lower(), urllib.parse.quote(v.strip(), safe=''))
            for k, v in headers.items() if k.lower() in signed_headers_set)
        canonical_headers = '\n'.join(f'{k}:{v}' for k, v in sorted_headers)
        signed_headers_str = ';'.join(sorted(signed_headers_set))
        return canonical_headers, signed_headers_str

    # 生成签名
    def generate_signature(self,
                           access_key,
                           secret_key,
                           region,
                           timestamp,
                           expiration_in_seconds,
                           method,
                           canonical_uri,
                           canonical_headers,
                           signed_headers_str,
                           CanonicalQueryString=''):
        signing_key_str = f'teleai-cloud-auth-v1/{access_key}/{region}/{timestamp}/{expiration_in_seconds}'
        signing_key = hmac.new(bytes(secret_key, 'utf-8'),
                               bytes(signing_key_str, 'utf-8'),
                               hashlib.sha256).hexdigest()
        canonical_request = f'{method.upper()}\n{canonical_uri}\n{CanonicalQueryString}\n{canonical_headers}'
        signature = hmac.new(bytes(signing_key, 'utf-8'),
                             bytes(canonical_request, 'utf-8'),
                             hashlib.sha256).hexdigest()
        authorization = f'{signing_key_str}/{signed_headers_str}/{signature}'
        return authorization

    def generate_signature_all(self,
                               access_key,
                               secret_key,
                               region,
                               timestamp,
                               expiration_in_seconds,
                               method,
                               canonical_uri,
                               headers,
                               CanonicalQueryString=''):
        canonical_headers, signed_headers_str = self.generate_canonical_headers(
            headers)
        return self.generate_signature(access_key, secret_key, region,
                                       timestamp, expiration_in_seconds,
                                       method, canonical_uri,
                                       canonical_headers, signed_headers_str,
                                       CanonicalQueryString)

    # 获取Content-Length
    def get_content_length(self, data):
        body = json.dumps(data)
        return str(len(body))
