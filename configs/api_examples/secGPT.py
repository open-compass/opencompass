# -*- coding: utf-8 -*-
import json
import re
from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://192.168.31.10:9997/v1"
#openai_api_base = "http://192.168.31.10:9997/v1"
#openai_api_base = "http://175.6.27.232:8811/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,

)

models = client.models.list()
model = models.data[0].id
print(model)

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
def ask_model(prompt, content, pp=False, temperature=0.1, top_p=1):
    ret = ""
    chat_completion = client.chat.completions.create(
        messages=[{
            "role": "system",
            "content": prompt,

        }, {
            "role": "user",
            "content": content+"回答内容以下格式：A"
        }, ],
        model=model,
        stream=True,
        max_tokens=4000,
        temperature=temperature,
        top_p=top_p,
        # stop=["<|endoftext|>","<|im_end|>"],
        # stop_token_ids=[2 ]
    )
    for chunk in chat_completion:
        i = chunk.choices[0].delta.content or ""
        ret += i
        if pp:
            print(i, end="")
    return ret


if __name__ == '__main__':
    from datasets import load_dataset

    dataset = load_dataset(r"cseval/cs-eval")
    submmit_json = {}
    for i in range(4669):  #13107
        text = dataset['test'][i]["prompt"]
        id = dataset['test'][i]["id"]
     #   print("******提问内容为：{}********\n".format(text))
      #  print("******提问id为：{}********\n".format(id))
        if "单选" in text:
            prompt = "你是一个考生，你需要根据考试内容，做出回答，只帮我选出唯一正确选项，以这种格式回答：A"

        if "多选" in text:
            prompt = "你是一个考生，你需要根据考试内容，做出回答，只帮我选出多个正确选项，以这种格式回答：A"
            ret = ask_model(prompt, text)

        else:
            prompt = "你是一个考生，会有中文英文问题，你要做出回答，只帮我选出正确选项的序号如：以这种格式回答：A"
            ret = ask_model(prompt, text)
      #  print("*****回答内容:：{}******** \n".format(ret))

        ret=clean_answer(ret)

      #  print("*****回答内容:：{}******** \n".format(ret))
        re_str = f'{{"question_id": "{id}", "answer": "{ret}"}}'
        with open(f"{model}.json", "a+", encoding="utf-8") as f:
            f.write(re_str + "," + "\n")
