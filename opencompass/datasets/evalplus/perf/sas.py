"""This file implements the "Synthesizing an Synthesizer" idea using OpenAI API.
Specifically, for each HumanEval+ and MBPP+ task, we generate n test input synthesizers
by querying a vLLM server (https://docs.vllm.ai/en/latest/).
"""

import json
from typing import Optional

import openai
from tqdm import tqdm

from evalplus.data import get_human_eval_plus, get_mbpp_plus


def fewshot_cot(
    task_id,
    client: openai.OpenAI,
    entry_point: str,
    code: str,
    model: str,
    n: int = 1,
    max_tokens: int = 2048,
):
    responses = client.completions.create(
        model=model,
        prompt=f'''\
You are an AI programming assistant, proficient in analyzing and generating Python code. \
You are going to produce a self-contained Python function to generate a large input for a given function, \
to test its performance at scale.
### Instruction:
Generate a `perf_input_gen(scale: int)` function to produce a "large" input to exercise the performance of the `add` function:
```python3
def add(x: int, y: int):
    """Add two numbers x and y
    >>> add(2, 3)
    5
    >>> add(5, 7)
    12
    """
    return x + y
```
### Response:
Analysis:
1. Input format: two integers `x` and `y`
2. Is this task O(1) solvable? Yes
### Instruction:
Generate a `perf_input_gen(scale: int)` function to produce a "large" input to exercise the performance of the `prime_num` function:
```python3
"""
Write a function to check if a number is prime or not.
assert prime_num(2) == True
"""
import math
def prime_num(num):
    if num < 2: return False
    for i in range(2, math.isqrt(num)):
        if num % i == 0:
            return False
    return True
```
### Response:
Analysis:
1. Input format: An integer `n`
2. Is this task O(1) solvable? No
3. Time complexity: O(n)
4. Space complexity: O(1)
5. What kind of input can exercise its performance? Large prime numbers
```python3
# Can reuse the `prime_num` function
# `scale` is a rough estimate of the input size -- larger `scale` means larger input
# use case: prime_num(*perf_input_gen(scale))
import random
def perf_input_gen(scale: int):
    for i in range(scale, 2, -1):
        if prime_num(i):
            return (i,)
    return (2,)
```
### Instruction:
Generate a `perf_input_gen(scale: int)` function to produce a "large" input to exercise the performance of the `{entry_point}` function:
```python3
{code}
```
### Response:
Analysis:
1. Input format: ''',
        n=n,
        stop=["\n```\n", "\n2. Is this task O(1) solvable? Yes"],
        max_tokens=max_tokens,
        temperature=0.2,
    )

    # warn if any response is out of context
    for r in responses.choices:
        if r.finish_reason == "length":
            print(f"Warning: response is too long for {task_id}")

    return [r.text for r in responses.choices]


def main(
    output: str,  # output file
    n: int = 16,  # sample size and batch size
    model: Optional[str] = "TheBloke/deepseek-coder-33B-instruct-AWQ",
    port: str = 8088,
):
    assert output.endswith(".jsonl"), "output must be a .jsonl file"

    base_url = f"http://localhost:{port}/v1"
    print(f"Trying to query vLLM model: {model} at {base_url}")
    print(f"Note: To use SaS, you need to first set up a vLLM server for {model}")
    print(f"For example:")
    print(
        f"""python -m vllm.entrypoints.openai.api_server \\
--model "{model}" \\
--port {port} \\
--tensor-parallel-size 2 \\
--max-num-seqs 16 \\
--gpu-memory-utilization 1.0"""
    )

    # "task_id" -> { "task_id", "entry_point", "ref_code", }
    tasks = {}
    for task_id, item in get_human_eval_plus().items():
        tasks[task_id] = {
            "task_id": task_id,
            "entry_point": item["entry_point"],
            "ref_code": item["prompt"] + item["canonical_solution"],
        }

    for task_id, item in get_mbpp_plus().items():
        tasks[task_id] = {
            "task_id": task_id,
            "entry_point": item["entry_point"],
            "ref_code": item["prompt"] + item["canonical_solution"],
        }

    # Using vLLM as a backend, please make sure that a vLLM server is available first.
    # vLLM document: https://docs.vllm.ai/en/latest/
    client = openai.OpenAI(api_key="none", base_url=base_url)

    with open(output, "w") as f:
        for task_id, item in tqdm(tasks.items(), total=len(tasks)):
            responses = fewshot_cot(
                task_id=task_id,
                client=client,
                entry_point=item["entry_point"],
                code=item["ref_code"],
                model=model,
                n=n,
            )
            f.write(
                json.dumps(
                    {
                        "task_id": task_id,
                        "ref_code": item["ref_code"],
                        "synthesizers": responses,
                    }
                )
                + "\n"
            )
            f.flush()


if __name__ == "__main__":
    import fire

    fire.Fire(main)
