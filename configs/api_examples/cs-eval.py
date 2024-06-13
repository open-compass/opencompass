# -*- coding: utf-8 -*-

import requests
import time

code="""
def get_ListId(data, key, key_vue, re_key):for i in data: if i[key] == key_vue：print(i[re_key]) return i[re_key] return 0
"""

ip_port = "http://175.6.27.232:8001"


def get_source(text):
    """
    供大模型生成内容的追溯功能，可展示大模型生成内容对应的源文件以及具体出处，大模型生成内容的追溯支持倒排序索引、embeding、bm25等搜索召回算法。

    :return:
    """

    url = "{}/chat".format(ip_port)
    # 这个方法是对应大模型生成内容的追溯
    prompts = [
        text
    ]
    # prompts = [
    #     '修复以下代码：{}'.format(code)
    # ]

    for prompt in prompts:
        data = {
            "context": prompt  # 问题
        }
        r = requests.post(url, stream=True, json=data)
        for line in r.iter_lines():
            response = line.decode('utf-8', 'ignore')  # 使用 UTF-8 编码并忽略不能解码的字符

        print(prompt, response)


def get_user_response():
    """
    提供用户反馈学习功能，支持用户对模型使用效果的评价
    :return:
    """
    url = "{}/vote".format(ip_port)
    data = {
        "response": "AI输出的结果",  # AI输出结果
        "type": 0,  # 0: 同意, 1: 反对, 2: 中立
        "argument": ""  # 用户编写的理由
    }
    r = requests.post(url, stream=True, json=data)
    print(r.json())


if __name__ == "__main__":

    from datasets import load_dataset
    dataset = load_dataset(r"cseval/cs-eval")
    for i in range(1000):
        text=dataset['test'][i]["prompt"]
        print("提问内容为：{}".format(text))
    # 内容追溯
     #   get_source(text)

    # 用户反馈
    # get_user_response()
