import ast
import random
from typing import List

import openai
from openai.types.chat import ChatCompletion

from evalplus.data.utils import to_raw
from evalplus.gen import BaseGen
from evalplus.gen.util import trusted_check_exec
from evalplus.gen.util.openai_request import make_auto_request


class ChatGPTGen(BaseGen):
    def __init__(self, inputs: List, signature: str, contract_code: str, gd_code: str):
        super().__init__(inputs, signature, contract_code)
        self.gd_code = gd_code
        self.prompt_messages = [
            "Please generate complex inputs to test the function.",
            "Please generate corner case inputs to test the function.",
            "Please generate difficult inputs to test the function.",
        ]
        self.iteration = 20
        self.client = openai.Client()

    def seed_selection(self) -> List:
        # get 5 for now.
        return random.sample(self.seed_pool, k=min(len(self.seed_pool), 5))

    @staticmethod
    def _parse_ret(ret: ChatCompletion) -> List:
        rets = []
        output = ret.choices[0].message.content
        if "```" in output:
            for x in output.split("```")[1].splitlines():
                if x.strip() == "":
                    continue
                try:
                    # remove comments
                    input = ast.literal_eval(f"[{x.split('#')[0].strip()}]")
                except:  # something wrong.
                    continue
                rets.append(input)
        return rets

    def chatgpt_generate(self, selected_inputs: List) -> List:
        # append the groundtruth function
        # actually it can be any function (maybe we can generate inputs for each llm generated code individually)
        message = f"Here is a function that we want to test:\n```\n{self.gd_code}\n```"
        str_inputs = "\n".join(
            [
                ", ".join([f"'{to_raw(i)}'" if type(i) == str else str(i) for i in x])
                for x in selected_inputs
            ]
        )
        message += f"\nThese are some example inputs used to test the function:\n```\n{str_inputs}\n```"
        message += f"\n{random.choice(self.prompt_messages)}"
        ret = make_auto_request(
            self.client,
            message=message,
            model="gpt-3.5-turbo",
            max_tokens=256,
            response_format={"type": "text"},
        )
        return self._parse_ret(ret)

    def generate(self, num: int):
        while len(self.new_inputs) < num and self.iteration >= 0:
            seeds = self.seed_selection()
            new_inputs = self.chatgpt_generate(seeds)
            for new_input in new_inputs:
                if hash(str(new_input)) not in self.seed_hash:
                    if trusted_check_exec(self.contract, [new_input], self.entry_point):
                        self.seed_pool.append(new_input)
                        self.seed_hash.add(hash(str(new_input)))
                        self.new_inputs.append(new_input)
            self.iteration -= 1
        return self.new_inputs[:num]
