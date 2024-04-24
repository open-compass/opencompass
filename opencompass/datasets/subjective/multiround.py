# flake8: noqa: E501
import json
import os.path as osp
import re
from typing import Optional

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET

from ..base import BaseDataset

base_prefix_en = """
You are a helper who will help me to evaluate the quality of AI assistants.
You will be provided with a dialogue between a user and an assistant of N rounds (each round includes an utterance from user and an utterance from the assistant).
In this dialogue, user will ask assistant to finish a series of tasks.
Your job is to help me detect whether the AI assistant successfully completed these tasks.
You should print a list of round index (starting from 1), in which rounds the assistants failed to accomplish the task assigned by user in the current round. Besides, you also need to provide the reason.
If the assistant succeed in the entire series of tasks, print an empty list '[]'.
Here I will show you some examples:

Example 1:
[{'round': 1, 'user': 'I have data about different car models that need to be formatted into a CSV. The data includes the car model, engine power in horsepower, and year of release. The cars are Mustang, 450hp, released in 2020; Corvette, 490hp, released in 2021; and Charger, 370hp, released in 2019. Could you create a CSV file for me?', 'assistant': 'Model,Engine Power (HP),Year of Release\nMustang,450,2020\nCorvette,490,2021\nCharger,370,2019'}, {'round': 2, 'user': 'Okay, now please add a new field for the body style of each car. Mustang is a Coupe, Corvette is a Coupe as well, and Charger is a Sedan.', 'assistant': 'Model,Engine Power (HP),Year of Release,Body Style\nMustang,450,2020,Coupe\nCorvette,490,2021,Coupe\nCharger,370,2019,Sedan'}]
Output: []
Reason: The assistant succeed in all round.

Example 2:
[{'round': 1, 'user': 'I have data about different car models that need to be formatted into a CSV. The data includes the car model, engine power in horsepower, and year of release. The cars are Mustang, 450hp, released in 2020; Corvette, 490hp, released in 2021; and Charger, 370hp, released in 2019. Could you create a CSV file for me?', 'assistant': 'Model,Engine Power (HP),Year of Release\nMustang,450,2020\nCorvette,490,2021\nCharger,370,2019'}, {'round': 2, 'user': 'Okay, now please add a new field for the body style of each car. Mustang is a Coupe, Corvette is a Coupe as well, and Charger is a Sedan.', 'assistant': 'Model,Engine Power (HP),Year of Release,Body Style\nMustang,450,2020,Coupe\nCorvette,490,2021,Coupe\nCharger,370,2019,Coupe'}]
Output: [2]
Reason: In round 2, the body style of Charger should be Sedan, but the assistant returned Coupe.

Example 3:
[{'round': 1, 'user': 'I have data about different car models that need to be formatted into a CSV. The data includes the car model, engine power in horsepower, and year of release. The cars are Mustang, 450hp, released in 2020; Corvette, 490hp, released in 2021; and Charger, 370hp, released in 2019. Could you create a CSV file for me?', 'assistant': 'Model,Engine Power (HP),Year of Release\nMustang,400,2020\nCorvette,400,2021\nCharger,400,2019'}, {'round': 2, 'user': 'Okay, now please add a new field for the body style of each car. Mustang is a Coupe, Corvette is a Coupe as well, and Charger is a Sedan.', 'assistant': 'Model,Engine Power (HP),Year of Release,Body Style\nMustang,450,2020,Coupe\nCorvette,490,2021,Coupe\nCharger,370,2019,Coupe'}]
Output: [1, 2]
Reason: In round 1, the horse power of all models are wrong. In round 2, the body style of Charger should be Sedan, but the assistant returned Coupe.

Now it's your turn:
Dialogue:
"""

base_suffix_en = """
Based on the dialogue, give your Output and Reason in the above format.
"""

base_prefix_ch = """
你是一个评估AI assistant对话内容质量的助手。
你将获得用户和与assistant的N轮对话（每轮包括用户的问题和assistant的回答）。
在这个对话中，用户会要求assistant完成一系列任务。
你需要检测assistant是否成功完成了这些任务。
你需要输出一个轮次索引的列表，列出assistant未能完成任务的轮次。此外，你还需要提供原因。
如果助手成功完成了整个系列任务，请打印一个空列表'[]'。
下面是一些例子：

例1：
[{'round': 1, 'user': '我有一些关于员工的数据，需要整理成 CSV 格式。数据包括员工的姓名、年龄和职位。员工是张伟，30岁，工程师；李娜，28岁，设计师；王浩，32岁，经理。请帮我制作一个 CSV 文件。', 'assistant': '姓名,年龄,职位\n张伟,30,工程师\n李娜,28,设计师\n王浩,32,经理'}, {'round': 2, 'user': '好的，现在请在每个员工信息后面增加一个薪水字段。张伟的薪水是8000，李娜是9000，王浩是12000。', 'assistant': '姓名,年龄,职位,薪水\n张伟,30,工程师,8000\n李娜,28,设计师,9000\n王浩,32,经理,12000'}]
输出：[]
原因：assistant成功完成了所有的任务。

例2：
[{'round': 1, 'user': '我有一些关于员工的数据，需要整理成 CSV 格式。数据包括员工的姓名、年龄和职位。员工是张伟，30岁，工程师；李娜，28岁，设计师；王浩，32岁，经理。请帮我制作一个 CSV 文件。', 'assistant': '姓名,年龄,职位\n张伟,30,工程师\n李娜,28,设计师\n王浩,32,经理'}, {'round': 2, 'user': '好的，现在请在每个员工信息后面增加一个薪水字段。张伟的薪水是8000，李娜是9000，王浩是12000。', 'assistant': '姓名,年龄,职位,薪水\n张伟,30,工程师,8000\n李娜,28,设计师,8000\n王浩,40,经理,12000'}]
输出：[2]
原因：在第二轮对话中，assistant错误的填写了李娜的薪水和王浩的年龄。

例3：
[{'round': 1, 'user': '我有一些关于员工的数据，需要整理成 CSV 格式。数据包括员工的姓名、年龄和职位。员工是张伟，30岁，工程师；李娜，28岁，设计师；王浩，32岁，经理。请帮我制作一个 CSV 文件。', 'assistant': '姓名,年龄,职位\n张伟,30,工程师\n张伟,28,设计师\n张伟,32,经理'}, {'round': 2, 'user': '好的，现在请在每个员工信息后面增加一个薪水字段。张伟的薪水是8000，李娜是9000，王浩是12000。', 'assistant': '姓名,年龄,职位,薪水\n张伟,30,工程师,8000\n李娜,28,设计师,8000\n王浩,40,经理,12000'}]
输出：[1, 2]
原因：在第一和第二轮对话中，assistant都出现了错误。

现在请评估以下对话：
对话：
"""

base_suffix_ch = """
基于以上对话，请按照上面的格式给出你的"输出"和"原因"。
"""


def prompt_construct(sample):
    if sample['others']['language'] == 'zh':
        return base_prefix_ch, base_suffix_ch
    elif sample['others']['language'] == 'en':
        return base_prefix_en, base_suffix_en
    # ref = str(sample['dialogue'])
    # base_suffix.format(ref=ref)


@LOAD_DATASET.register_module()
class MultiroundDataset(BaseDataset):

    def load(
        self,
        path: str,
        name: str,
    ):
        filename = osp.join(path, f'{name}.json')
        dataset = DatasetDict()
        raw_data = []
        with open(filename, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            for problem in json_data:
                gpt4_prefix, gpt4_suffix = prompt_construct(problem)
                dialogue = problem['dialogue']
                capability = str(problem['capability'])
                others = problem['others']
                others['round'] = int(len(dialogue) / 2)
                raw_data.append({
                    'dialogue': dialogue,
                    'capability': capability,
                    'gpt4_prefix': gpt4_prefix,
                    'gpt4_suffix': gpt4_suffix,
                    'others': others,
                    'judge': {
                        'capability': capability,
                        'others': others
                    }
                })
        dataset = Dataset.from_list(raw_data)
        return dataset
