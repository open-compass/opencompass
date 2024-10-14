import json
import time
#
# id=2
# ret=3
# re = '{"id": ' + str(id) + ', "ret": ' + str(ret) + '}'
# print(re)
#
#
# # json_text='{"type": "chat", "content": "B"}'
# #
# # json_text=json.loads(json_text)
# # print(json_text)
#
#
# import re  # 导入正则表达式模块
#
#
# def clean_answer(answer):
#     # 使用正则表达式编译模式
#     pattern = re.compile(r'[ABCD对错]')
#     # 使用 finditer 查找所有不重复的匹配项
#     matches = pattern.finditer(answer)
#     # 从每个匹配项中提取字符，并确保每个字符只出现一次
#     cleaned_answer = ""
#     for match in matches:
#         # 检查字符是否已经添加到结果中
#         if match.group(0) not in cleaned_answer:
#             cleaned_answer += match.group(0)
#     return cleaned_answer
#
# # 示例使用
# answer = "答案：ABCDAB对错"
# print(clean_answer(answer))  # 应该输出 "BCD对错A"
#
#
# def get_json_time(day):
#     if day == 0:
#         return int(round(time.time()))  # 如果是零返回今天时间戳
#     else:
#         now = int(round(time.time()))
#         return now + 3600 * 24 * day
# # 示例使用
# print(f"今天的时间戳为：{get_json_time(0)}")  # 输出今天的时间戳)



import requests
import json

# API endpoint
url = 'http://192.168.31.69:10081/v1/chat-messages'

# Headers
headers = {
    'Authorization': 'Bearer app-AN8ZtFLNp6IR6tXl6aylTteS',
    'Content-Type': 'application/json'
}

# Data payload
# data = {
#     "inputs": "",
#    "query": "你是谁",
#     "response_mode": "streaming",
#     "conversation_id": "",
#     "user": "abc-123",
# }

data = {
    "inputs": "",
   "query": "这句话是什么意思：Traditional network toward return party. Officer push system race. Miss no station past often from young.",
    "response_mode": "streaming",
    "conversation_id": "",
    "user": "abc-123",
}

# Convert data to JSON format
json_data = json.dumps(data)

# Send POST request
response = requests.post(url, headers=headers, data=json_data)

# Check if the request was successful
if response.status_code == 200:
    print("Request was successful.")
    print("Response:", response.text)
else:
    print("Request failed with status code:", response.status_code)
    print("Response:", response.text)