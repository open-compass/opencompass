# -*- coding: utf-8 -*-
import json
import re
import argparse

from openai import OpenAI
from tqdm import tqdm  # 导入tqdm

# 解析命令行参数
parser = argparse.ArgumentParser(description="Run the model testing script.")
parser.add_argument("model", type=str, help="The model ID to use.")
parser.add_argument("url", type=str, help="The IP address of the API server.")

args = parser.parse_args()

# 使用命令行参数设置 API key 和 API base
openai_api_key = "EMPTY"
openai_api_base = f"{args.url}"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = args.model  # 使用命令行参数中的模型ID
print(model)

def clean_answer(answer):
    pattern = re.compile(r'[ABCD对错]')
    matches = pattern.finditer(answer)
    cleaned_answer = ""
    for match in matches:
        if match.group(0) not in cleaned_answer:
            cleaned_answer += match.group(0)
    return cleaned_answer

def ask_model(prompt, content, pp=False, temperature=0.1, top_p=1):
    ret = ""
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": content + "回答内容以下格式：A"}
            ],
            model=model,
            stream=True,
            max_tokens=4000,
            temperature=temperature,
            top_p=top_p,
        )
        for chunk in chat_completion:
            i = chunk.choices[0].delta.content or ""
            ret += i
            if pp:
                print(i, end="")
    except Exception as e:
        print(f"Error in ask_model: {e}")
        ret = None
    return ret

if __name__ == '__main__':
    from datasets import load_dataset
    from tqdm import tqdm

    dataset = load_dataset(r"cseval/cs-eval")
    total_questions = len(dataset['test'])  # 获取测试集中的问题总数
    json_data = []  # 初始化一个列表来存储所有问题的 JSON 数据

    pbar = tqdm(total=4670, desc='Processing questions')
    for i in range(4670):
        text = dataset['test'][i]["prompt"]
        id = dataset['test'][i]["id"]
        if "单选" in text:
            prompt = "你是一个考生，你需要根据考试内容，做出回答，只帮我选出唯一正确选项，以这种格式回答：A"

        if "多选" in text:
            prompt = "你是一个考生，你需要根据考试内容，做出回答，只帮我选出多个正确选项，以这种格式回答：A"
            ret = ask_model(prompt, text)

        else:
            prompt = "你是一个考生，会有中文英文问题，你要做出回答，只帮我选出正确选项的序号如：以这种格式回答：A"
            ret = ask_model(prompt, text)
            #  print("*****回答内容:：{}******** \n".format(ret))

        ret = clean_answer(ret)

        #  print("*****回答内容:：{}******** \n".format(ret))
        re_str = f'{{"question_id": "{id}", "answer": "{ret}"}}'
        with open(f"{model}.json", "a+", encoding="utf-8") as f:
            f.write(re_str + "," + "\n")
        pbar.update(1)  # 更新进度条
    pbar.close()  # 确保进度条正确关闭



    print("Script executed successfully.")
