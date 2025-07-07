import argparse
import json
from typing import Dict, List, Optional, Union

import requests


class LarkReporter:

    def __init__(self, url: str):
        self.url = url

    def post(self,
             content: Union[str, List[List[Dict]]],
             title: Optional[str] = None):
        """Post a message to Lark.

        When title is None, message must be a str.
        otherwise msg can be in rich text format (see
        https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/im-v1/message/create_json#45e0953e
        for details).
        """
        if title is None:
            assert isinstance(content, str)
            msg = {'msg_type': 'text', 'content': {'text': content}}
        else:
            if isinstance(content, str):
                content = [[{'tag': 'text', 'text': content}]]
            msg = {
                'msg_type': 'post',
                'content': {
                    'post': {
                        'zh_cn': {
                            'title': title,
                            'content': content
                        }
                    }
                }
            }
        requests.post(self.url, data=json.dumps(msg))


def parse_args():
    parser = argparse.ArgumentParser(description='Lark bot reporter')
    parser.add_argument('url', help='Lark bot url')
    parser.add_argument('content', type=str, help='Content')
    parser.add_argument('--title', type=str, help='Title', default=None)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    lark = LarkReporter(args.url)
    if args.title:
        lark.post(args.content, args.title)
    else:
        lark.post(args.content)
