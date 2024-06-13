import json

id=2
ret=3
re = '{"id": ' + str(id) + ', "ret": ' + str(ret) + '}'
print(re)


# json_text='{"type": "chat", "content": "B"}'
#
# json_text=json.loads(json_text)
# print(json_text)


import re  # 导入正则表达式模块


def clean_answer(answer):
    # 使用正则表达式编译模式
    pattern = re.compile(r'[ABCD对错]')
    # 使用 finditer 查找所有不重复的匹配项
    matches = pattern.finditer(answer)
    # 从每个匹配项中提取字符，并确保每个字符只出现一次
    cleaned_answer = ""
    for match in matches:
        # 检查字符是否已经添加到结果中
        if match.group(0) not in cleaned_answer:
            cleaned_answer += match.group(0)
    return cleaned_answer

# 示例使用
answer = "答案：ABCDAB对错"
print(clean_answer(answer))  # 应该输出 "BCD对错A"