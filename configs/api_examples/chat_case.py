# -*- coding: utf-8 -*-
import json

import requests
import time

code = """
def get_ListId(data, key, key_vue, re_key):for i in data: if i[key] == key_vue：print(i[re_key]) return i[re_key] return 0
"""

ip_port = "http://175.6.27.232:8001"


def get_source(prompt):
    """
    供大模型生成内容的追溯功能，可展示大模型生成内容对应的源文件以及具体出处，大模型生成内容的追溯支持倒排序索引、embeding、bm25等搜索召回算法。

    :return:
    """

    url = "{}/chat".format(ip_port)
    # 这个方法是对应大模型生成内容的追溯
    # prompts = [
    #     "多选题：\n\n在网络安全的AI应用中，如何确保AI系统不被恶意利用？（多选）  \nA. 强化模型对抗性攻击的防御 \nB. 定期进行模型的安全评估 \nC. 限制模型训练数据的来源 \nD. 实现AI决策过程的全面监控和记录\n\n请给出符合题意的所有选项。"
    # ]
    # # prompts = [
    # #     '修复以下代码：{}'.format(code)
    # # ]

    data = {
        "context": prompt+"回答内容，只回答选项，例如：A"  # 问题
    }
    for attempt in range(3):
        try:
            r = requests.post(url, stream=True, json=data,timeout=300)
            for line in r.iter_lines():
                response = line.decode('utf-8', 'ignore')  # 使用 UTF-8 编码并忽略不能解码的字符

            print(prompt, response)
            answer=json.loads(response)
            return answer["content"]
        except requests.exceptions.Timeout:
            print(f"请求超时，尝试次数 {attempt + 1}/{3}，将在 {30} 秒后重试...")
            time.sleep(30)
        except requests.exceptions.RequestException as e:
            # 处理其他请求异常
            print(f"请求失败：{e}")
            break
    return None  # 如果重试次数用完，返回 None
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
    print(r.json()["content"])


if __name__ == "__main__":
    from datasets import load_dataset

    dataset = load_dataset(r"cseval/cs-eval")
    for i in range(13108):  # 13107
        text = dataset['test'][i]["prompt"]
        id = dataset['test'][i]["id"]

        answer=get_source(text)
        print("回答内容：{}".format(answer))
        re_str = f'{{"question_id": "{id}", "answer": "{answer}"}}'
        with open("chat_submit.json", "a+", encoding="utf-8") as f:
            f.write(re_str + "," + "\n")
    # 用户反馈
    # get_user_response()
