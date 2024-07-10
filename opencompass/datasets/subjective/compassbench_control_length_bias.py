# flake8: noqa
import json
import os.path as osp

from datasets import Dataset

from opencompass.registry import LOAD_DATASET

from ..base import BaseDataset

base_prompt_zh = """
请根据 用户问题 以及 相应的两个回答，判断哪一个回答更好。
[用户问题]
{question}

[回答1开始]
{prediction}
[回答1结束]

回答1中的中文字符数：{prediction_cn_word_count}
回答1中的英文单词数：{prediction_en_word_count}

[回答2开始]
{prediction2}
[回答2结束]

回答2中的中文字符数：{prediction2_cn_word_count}
回答2中的英文单词数：{prediction2_en_word_count}

请注意：
1. 若题目中有明确字数限制，打分时应该将字数纳入考虑。如果回答超出或少于规定的字数限制，应相应扣分。
2. 在没有字数限制的情况下，回答的简洁性和直接性应被优先考虑，除非详细程度对于理解答案至关重要。
3. 如果两个回答都准确地解决了用户的问题，但一个回答更加简洁，而另一个回答提供了不必要的额外信息，那么简洁的回答可能会得到更高的评分。
4. 在评分时，还应考虑回答的可读性和用户友好性，例如代码的清晰度和注释的充分性。

请先对两个回答进行评价，最后在以下 3 个选项中做出选择:
A. 回答1更好
B. 回答2更好
C. 回答1、2平局

如果你认为回答1更好，你的输出应形如：
评价1：回答1 xxx
评价2：回答2 xxx
选择：[[A]]

如果你认为回答2更好，你的输出应形如：
评价1：回答1 xxx
评价2：回答2 xxx
选择：[[B]]

如果你认为回答1、2打成平手，你的输出应形如：
评价1：回答1 xxx
评价2：回答2 xxx
选择：[[C]]
""".strip()

base_prompt_en = """
Please evaluate the two responses based on the user's question and then choose from the following three options:
A. Response 1 is better
B. Response 2 is better
C. Both responses are equal

[user's question]
{question}

[Response 1 Start]
{prediction}
[Response 1 End]

The number of Chinese characters in Response 1: {prediction_cn_word_count}
The number of English words in Response 1: {prediction_en_word_count}

[Response 2 Start]
{prediction2}
[Response 2 End]

The number of Chinese characters in Response 2: {prediction2_cn_word_count}
The number of English words in Response 2: {prediction2_en_word_count}

Note:

1. If there is a clear word limit in the question, the word count should be taken into consideration when scoring. If the answer exceeds or falls short of the specified word limit, points should be deducted accordingly.
2. In the absence of a word limit, the conciseness and directness of the answer should be given priority, unless the level of detail is essential for understanding the answer.
3. If both answers accurately solve the user's question, but one is more concise and the other provides unnecessary additional information, the concise answer may receive a higher score.
4. When scoring, the readability and user-friendliness of the answer should also be considered, such as the clarity of the code and the adequacy of the comments.


If you believe that Response 1 is better, your output should be formatted as follows:
Evaluation 1: Response 1 xxx
Evaluation 2: Response 2 xxx
Choice: [[A]]

If you believe that Response 2 is better, your output should be formatted as follows:
Evaluation 1: Response 1 xxx
Evaluation 2: Response 2 xxx
Choice: [[B]]

If you believe that both responses are equally good, your output should be formatted as follows:
Evaluation 1: Response 1 xxx
Evaluation 2: Response 2 xxx
Choice: [[C]]
""".strip()


@LOAD_DATASET.register_module()
class CompassBenchControlLengthBiasDataset(BaseDataset):

    def load(self, path: str, name: str, *args, **kwargs):
        filename = osp.join(path, f'{name}.json')
        raw_data = []
        with open(filename, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            for problem in json_data:
                question = problem['question']
                lan = problem['language']
                others = problem['others']
                judge_prompt = base_prompt_zh if lan == 'zh' else base_prompt_en
                judge_prompt = judge_prompt.replace('{question}', question)
                raw_data.append({
                    'question': question,
                    'judge_prompt': judge_prompt,
                    'judge': {
                        'lan': lan,
                        'level': others['level'],
                        'category': problem['category'],
                        'question': question
                    }
                })
        dataset = Dataset.from_list(raw_data)
        return dataset
