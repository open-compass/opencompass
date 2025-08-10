from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import HumanevalDataset, HumanEvalEvaluator, HumanEvalPlusEvaluator, humaneval_postprocess_v2
from opencompass.datasets import MBPPDataset, SanitizedMBPPDataset, MBPPEvaluator
from opencompass.datasets import HumanevalXDataset, HumanevalXEvaluator
from opencompass.datasets import LCDataset, LCPassKEvaluator
from opencompass.datasets import TACODataset, TACOEvaluator

compassbench_v1_1_code_datasets = []

# --------------------------------------------------------------- HumanEval CN ---------------------------------------------------------------
humaneval_reader_cfg = dict(input_columns=['prompt'], output_column='task_id', train_split='test')

humaneval_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='完成以下Python代码任务:\n{prompt}'),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512),
)

humaneval_eval_cfg = dict(
    evaluator=dict(type=HumanEvalEvaluator),
    pred_postprocessor=dict(type=humaneval_postprocess_v2),
)

compassbench_v1_1_code_datasets.append(
    dict(
        abbr='openai_humaneval_cn',
        type=HumanevalDataset,
        path='data/compassbench_v1.1/code/humaneval_cn/human-eval-cn-v2-20210705.jsonl',
        reader_cfg=humaneval_reader_cfg,
        infer_cfg=humaneval_infer_cfg,
        eval_cfg=humaneval_eval_cfg,
    )
)

# --------------------------------------------------------------- HumanEval Plus ---------------------------------------------------------------
humaneval_plus_reader_cfg = dict(input_columns=['prompt'], output_column='task_id', train_split='test')

# TODO: allow empty output-column
humaneval_plus_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='Complete the following python code:\n{prompt}'),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512),
)

humaneval_plus_eval_cfg = dict(
    evaluator=dict(type=HumanEvalPlusEvaluator),
    pred_postprocessor=dict(type=humaneval_postprocess_v2),
)

compassbench_v1_1_code_datasets.append(
    dict(
        abbr='humaneval_plus',
        type=HumanevalDataset,
        path='data/compassbench_v1.1/code/humaneval/human-eval-v2-20210705.jsonl',
        reader_cfg=humaneval_plus_reader_cfg,
        infer_cfg=humaneval_plus_infer_cfg,
        eval_cfg=humaneval_plus_eval_cfg,
    )
)

# --------------------------------------------------------------- MBPP CN ---------------------------------------------------------------
mbpp_reader_cfg = dict(input_columns=['text', 'test_list'], output_column='test_list_2')

mbpp_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='你是一名专业的 Python 程序员，你的任务是：编写一个函数，从给定的两个元组列表中查找相似的元素。 你的代码应该通过这些测试：\n\n assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)\n assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4) \n assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14) \n'),
                dict(role='BOT', prompt="[BEGIN]\n 'def similar_elements(test_tup1, test_tup2):\r\n  res = tuple(set(test_tup1) & set(test_tup2))\r\n  return (res)' \n[DONE] \n\n "),
                dict(role='HUMAN', prompt='你是一名专业的 Python 程序员，你的任务是：编写一个 Python 函数来识别一个整数是否不是素数。 你的代码应该通过这些测试：\n\n assert is_not_prime(2) == False \n assert is_not_prime(10) == True \n assert is_not_prime(35) == True \n'),
                dict(role='BOT', prompt="[BEGIN]\n 'import math\r\ndef is_not_prime(n):\r\n    result = False\r\n    for i in range(2,int(math.sqrt(n)) + 1):\r\n        if n % i == 0:\r\n            result = True\r\n    return result' \n[DONE] \n\n "),
                dict(role='HUMAN', prompt='你是一名专业的 Python 程序员，你的任务是：编写一个函数，使用堆队列算法从给定的数字列表中查找最大整数。 你的代码应该通过这些测试：\n\n assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] \n assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] \n assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35] \n'),
                dict(role='BOT', prompt="[BEGIN]\n 'import heapq as hq\r\ndef heap_queue_largest(nums,n):\r\n  largest_nums = hq.nlargest(n, nums)\r\n  return largest_nums' \n[DONE] \n\n "),
                dict(role='HUMAN', prompt='你是一名专业的 Python 程序员，你的任务是: {text} 你的代码应该通过这些测试:\n\n {test_list}  \n'),
                dict(role='BOT', prompt='[BEGIN]\n'),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512),
)

mbpp_eval_cfg = dict(evaluator=dict(type=MBPPEvaluator), pred_role='BOT')

compassbench_v1_1_code_datasets.append(
    dict(
        type=MBPPDataset,
        abbr='mbpp_cn',
        path='data/compassbench_v1.1/code/mbpp_cn/mbpp_cn.jsonl',
        reader_cfg=mbpp_reader_cfg,
        infer_cfg=mbpp_infer_cfg,
        eval_cfg=mbpp_eval_cfg,
    )
)

# --------------------------------------------------------------- Sanitized MBPP ---------------------------------------------------------------
sanitized_mbpp_reader_cfg = dict(input_columns=['text', 'test_list'], output_column='test_list_2')

sanitized_mbpp_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='You are an expert Python programmer, and here is your task: Write a function to find the similar elements from the given two tuple lists. Your code should pass these tests:\n\n assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)\n assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4) \n assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14) \n'),
                dict(role='BOT', prompt="[BEGIN]\n 'def similar_elements(test_tup1, test_tup2):\r\n  res = tuple(set(test_tup1) & set(test_tup2))\r\n  return (res)' \n[DONE] \n\n "),
                dict(role='HUMAN', prompt='You are an expert Python programmer, and here is your task: Write a python function to identify non-prime numbers. Your code should pass these tests:\n\n assert is_not_prime(2) == False \n assert is_not_prime(10) == True \n assert is_not_prime(35) == True \n'),
                dict(role='BOT', prompt="[BEGIN]\n 'import math\r\ndef is_not_prime(n):\r\n    result = False\r\n    for i in range(2,int(math.sqrt(n)) + 1):\r\n        if n % i == 0:\r\n            result = True\r\n    return result' \n[DONE] \n\n "),
                dict(role='HUMAN', prompt='You are an expert Python programmer, and here is your task: Write a function to find the largest integers from a given list of numbers using heap queue algorithm. Your code should pass these tests:\n\n assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] \n assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] \n assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35] \n'),
                dict(role='BOT', prompt="[BEGIN]\n 'import heapq as hq\r\ndef heap_queue_largest(nums,n):\r\n  largest_nums = hq.nlargest(n, nums)\r\n  return largest_nums' \n[DONE] \n\n "),
                dict(role='HUMAN', prompt='You are an expert Python programmer, and here is your task: {text} Your code should pass these tests:\n\n {test_list}  \n'),
                dict(role='BOT', prompt='[BEGIN]\n'),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512),
)

sanitized_mbpp_eval_cfg = dict(evaluator=dict(type=MBPPEvaluator), pred_role='BOT')

compassbench_v1_1_code_datasets.append(
    dict(
        type=SanitizedMBPPDataset,
        abbr='sanitized_mbpp',
        path='data/compassbench_v1.1/code/mbpp/sanitized-mbpp.jsonl',
        reader_cfg=sanitized_mbpp_reader_cfg,
        infer_cfg=sanitized_mbpp_infer_cfg,
        eval_cfg=sanitized_mbpp_eval_cfg,
    )
)

# --------------------------------------------------------------- HumanevalX ---------------------------------------------------------------
humanevalx_reader_cfg = dict(input_columns=['prompt'], output_column='declaration', train_split='test')

humanevalx_infer_cfg = dict(
    prompt_template=dict(type=PromptTemplate, template='{prompt}'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=1024),
)

humanevalx_eval_cfg_dict = {
    lang: dict(
        evaluator=dict(
            type=HumanevalXEvaluator,
            language=lang,
            ip_address=
            'localhost',  # replace to your code_eval_server ip_address, port
            port=5001,
        ),  # refer to https://opencompass.readthedocs.io/en/latest/advanced_guides/code_eval_service.html to launch a server
    )
    for lang in ['python', 'cpp', 'go', 'java', 'js']  # do not support rust now
}

# Please download the needed `xx.jsonl.gz` from
# https://github.com/THUDM/CodeGeeX2/tree/main/benchmark/humanevalx
# and move them into `data/humanevalx/` folder
for lang in ['python', 'cpp', 'go', 'java', 'js']:
    compassbench_v1_1_code_datasets.append(
        dict(
            type=HumanevalXDataset,
            abbr=f'humanevalx-{lang}',
            language=lang,
            path='data/compassbench_v1.1/code/humanevalx',
            reader_cfg=humanevalx_reader_cfg,
            infer_cfg=humanevalx_infer_cfg,
            eval_cfg=humanevalx_eval_cfg_dict[lang],
        )
    )

# --------------------------------------------------------------- LCBench ---------------------------------------------------------------
LC_difficulties_list = ['EASY', 'MEDIUM', 'HARD']
LC_reader_cfg = dict(input_columns=['text', 'test_list'], output_column='test_column')


LC_en_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='You are an expert Python programmer, and here is your task: You are given three positive integers n, x, and y.\nIn a city, there exist houses numbered 1 to n connected by n streets. There is a street connecting the house numbered i with the house numbered i + 1 for all 1 <= i <= n - 1 . An additional street connects the house numbered x with the house numbered y.\nFor each k, such that 1 <= k <= n, you need to find the number of pairs of houses (house1, house2) such that the minimum number of streets that need to be traveled to reach house2 from house1 is k.\nReturn a 1-indexed array result of length n where result[k] represents the total number of pairs of houses such that the minimum streets required to reach one house from the other is k.\nNote that x and y can be equal. Your code should pass these tests:\n\n assert countOfPairs(n = 3, x = 1, y = 3) == [6,0,0]\n assert countOfPairs(n = 5, x = 2, y = 4) == [10,8,2,0,0] \n assert countOfPairs(n = 4, x = 1, y = 1) == [6,4,2,0] \n'),
                dict(role='BOT', prompt="[BEGIN]\n 'from itertools import accumulate\ndef countOfPairs(n, x, y):\n        x, y = min(x, y), max(x, y)\n        A = [0] * n\n        for i in range(1, n + 1):\n            A[0] += 2                                   \n            A[min(i - 1, abs(i - y) + x)] -= 1          \n            A[min(n - i, abs(i - x) + 1 + n - y)] -= 1  \n            A[min(abs(i - x), abs(y - i) + 1)] += 1     \n            A[min(abs(i - x) + 1, abs(y - i))] += 1     \n            r = max(x - i, 0) + max(i - y, 0)\n            A[r + (y - x + 0) // 2] -= 1                \n            A[r + (y - x + 1) // 2] -= 1                \n        return list(accumulate(A))' \n[DONE] \n\n "),
                dict(role='HUMAN', prompt='You are an expert Python programmer, and here is your task: You are given a string word containing lowercase English letters.\nTelephone keypads have keys mapped with distinct collections of lowercase English letters, which can be used to form words by pushing them. For example, the key 2 is mapped with ["a","b","c"], we need to push the key one time to type "a", two times to type "b", and three times to type "c" .\nIt is allowed to remap the keys numbered 2 to 9 to distinct collections of letters. The keys can be remapped to any amount of letters, but each letter must be mapped to exactly one key. You need to find the minimum number of times the keys will be pushed to type the string word.\nReturn the minimum number of pushes needed to type word after remapping the keys.\nAn example mapping of letters to keys on a telephone keypad is given below. Note that 1, *, #, and 0 do not map to any letters. Your code should pass these tests:\n\n assert minimumPushes("abcde") == 5 \n assert minimumPushes("xyzxyzxyzxyz") == 12 \n assert minimumPushes("aabbccddeeffgghhiiiiii") == 24 \n'),
                dict(role='BOT', prompt="[BEGIN]\n 'def minimumPushes(word):\n        letter_counts = {}\n        for c in word:\n            letter_counts[c] = letter_counts.get(c, 0) + 1\n        counts = list(letter_counts.values())\n        counts.sort(reverse=True)\n        ans, row = 0, 1\n        for i in range(len(counts)):\n            if i > 7 and i % 8 == 0:\n                row += 1\n            ans += row * counts[i]\n        return ans' \n[DONE] \n\n "),
                dict(role='HUMAN', prompt="You are an expert Python programmer, and here is your task: You are given an integer k and an integer x.\nConsider s is the 1-indexed binary representation of an integer num. The price of a number num is the number of i's such that i % x == 0 and s[i] is a set bit.\nReturn the greatest integer num such that the sum of prices of all numbers from 1 to num is less than or equal to k.\nNote:\nIn the binary representation of a number set bit is a bit of value 1.\nThe binary representation of a number will be indexed from right to left. For example, if s == 11100, s[4] == 1 and s[2] == 0. Your code should pass these tests:\n\n assert findMaximumNumber(k = 9, x = 1) == 6 \n assert findMaximumNumber(k = 7, x = 2) == 9 \n"),
                dict(role='BOT', prompt="[BEGIN]\n 'def findMaximumNumber(k, x):\n        def check(v):\n            A = list(map(int, bin(v)[2:]))\n            n = len(A)\n            res = p = 0\n            for i,v in enumerate(A):\n                if v == 1:\n                    l = n - i - 1\n                    res += (p << l) + ((l // x) << (l - 1) if l else 0)\n                if (n - i) % x == 0:\n                    p += v\n            return res + p\n        l, r = 1, 10 ** 15\n        while l < r:\n            mid = (l + r + 1) // 2\n            if check(mid) <= k:\n                l = mid\n            else:\n                r = mid - 1\n        return l' \n[DONE] \n\n "),
                dict(role='HUMAN', prompt='You are an expert Python programmer, and here is your task: {text} Your code should pass these tests:\n\n {test_list}  \n'),
                dict(role='BOT', prompt='[BEGIN]\n'),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512),
)


LC_cn_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='你是一名专业的 Python 程序员，下面是你的任务： 给你三个 正整数 n 、x 和 y 。\n在城市中，存在编号从 1 到 n 的房屋，由 n 条街道相连。对所有 1 <= i < n ，都存在一条街道连接编号为 i 的房屋与编号为 i + 1 的房屋。另存在一条街道连接编号为 x 的房屋与编号为 y 的房屋。\n对于每个 k（1 <= k <= n），你需要找出所有满足要求的 房屋对 [house1, house2] ，即从 house1 到 house2 需要经过的 最少 街道数为 k 。\n返回一个下标从 1 开始且长度为 n 的数组 result ，其中 result[k] 表示所有满足要求的房屋对的数量，即从一个房屋到另一个房屋需要经过的 最少 街道数为 k 。\n注意，x 与 y 可以 相等。你的代码需要通过以下测试:\n\n assert countOfPairs(n = 3, x = 1, y = 3) == [6,0,0]\n assert countOfPairs(n = 5, x = 2, y = 4) == [10,8,2,0,0] \n assert countOfPairs(n = 4, x = 1, y = 1) == [6,4,2,0] \n'),
                dict(role='BOT', prompt="[BEGIN]\n 'from itertools import accumulate\ndef countOfPairs(n, x, y):\n        x, y = min(x, y), max(x, y)\n        A = [0] * n\n        for i in range(1, n + 1):\n            A[0] += 2                                   \n            A[min(i - 1, abs(i - y) + x)] -= 1          \n            A[min(n - i, abs(i - x) + 1 + n - y)] -= 1  \n            A[min(abs(i - x), abs(y - i) + 1)] += 1     \n            A[min(abs(i - x) + 1, abs(y - i))] += 1     \n            r = max(x - i, 0) + max(i - y, 0)\n            A[r + (y - x + 0) // 2] -= 1                \n            A[r + (y - x + 1) // 2] -= 1                \n        return list(accumulate(A))' \n[DONE] \n\n "),
                dict(role='HUMAN', prompt='你是一名专业的 Python 程序员，下面是你的任务: 给你一个字符串 word，由 不同 小写英文字母组成。\n电话键盘上的按键与 不同 小写英文字母集合相映射，可以通过按压按键来组成单词。例如，按键 2 对应 ["a","b","c"]，我们需要按一次键来输入 "a"，按两次键来输入 "b"，按三次键来输入 "c"。\n现在允许你将编号为 2 到 9 的按键重新映射到 不同 字母集合。每个按键可以映射到 任意数量 的字母，但每个字母 必须 恰好 映射到 一个 按键上。你需要找到输入字符串 word 所需的 最少 按键次数。\n返回重新映射按键后输入 word 所需的 最少 按键次数。\n下面给出了一种电话键盘上字母到按键的映射作为示例。注意 1，*，# 和 0 不 对应任何字母。你的代码需要通过以下测试:\n\n assert minimumPushes("abcde") == 5 \n assert minimumPushes("xyzxyzxyzxyz") == 12 \n assert minimumPushes("aabbccddeeffgghhiiiiii") == 24 \n'),
                dict(role='BOT', prompt="[BEGIN]\n 'def minimumPushes(word):\n        letter_counts = {}\n        for c in word:\n            letter_counts[c] = letter_counts.get(c, 0) + 1\n        counts = list(letter_counts.values())\n        counts.sort(reverse=True)\n        ans, row = 0, 1\n        for i in range(len(counts)):\n            if i > 7 and i % 8 == 0:\n                row += 1\n            ans += row * counts[i]\n        return ans' \n[DONE] \n\n "),
                dict(role='HUMAN', prompt='你是一名专业的 Python 程序员，下面是你的任务: 给你一个整数 k 和一个整数 x 。\n令 s 为整数 num 的下标从 1 开始的二进制表示。我们说一个整数 num 的 价值 是满足 i % x == 0 且 s[i] 是 设置位 的 i 的数目。\n请你返回 最大 整数 num ，满足从 1 到 num 的所有整数的 价值 和小于等于 k 。\n注意：\n一个整数二进制表示下 设置位 是值为 1 的数位。\n一个整数的二进制表示下标从右到左编号，比方说如果 s == 11100 ，那么 s[4] == 1 且 s[2] == 0。你的代码需要通过以下测试:\n\n assert findMaximumNumber(k = 9, x = 1) == 6 \n assert findMaximumNumber(k = 7, x = 2) == 9 \n'),
                dict(role='BOT', prompt="[BEGIN]\n 'def findMaximumNumber(k, x):\n        def check(v):\n            A = list(map(int, bin(v)[2:]))\n            n = len(A)\n            res = p = 0\n            for i,v in enumerate(A):\n                if v == 1:\n                    l = n - i - 1\n                    res += (p << l) + ((l // x) << (l - 1) if l else 0)\n                if (n - i) % x == 0:\n                    p += v\n            return res + p\n        l, r = 1, 10 ** 15\n        while l < r:\n            mid = (l + r + 1) // 2\n            if check(mid) <= k:\n                l = mid\n            else:\n                r = mid - 1\n        return l' \n[DONE] \n\n "),
                dict(role='HUMAN', prompt='你是一名专业的 Python 程序员，下面是你的任务: {text} 你的代码需要通过以下测试:\n\n {test_list}  \n'),
                dict(role='BOT', prompt='[BEGIN]\n'),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512),
)


LC_eval_cfg = dict(evaluator=dict(type=LCPassKEvaluator), pred_role='BOT')

for difficulty in LC_difficulties_list:
    compassbench_v1_1_code_datasets.append(
        dict(
            type=LCDataset,
            abbr='lcbench_en-' + difficulty,
            path='data/compassbench_v1.1/code/LCBench2023/LCBench2023.jsonl',
            difficulty=difficulty,
            reader_cfg=LC_reader_cfg,
            infer_cfg=LC_en_infer_cfg,
            eval_cfg=LC_eval_cfg,
        )
    )
    compassbench_v1_1_code_datasets.append(
        dict(
            type=LCDataset,
            abbr='lcbench_cn-' + difficulty,
            path='data/compassbench_v1.1/code/LCBench2023/LCBench2023_cn.jsonl',
            difficulty=difficulty,
            reader_cfg=LC_reader_cfg,
            infer_cfg=LC_cn_infer_cfg,
            eval_cfg=LC_eval_cfg,
        )
    )


# --------------------------------------------------------------- TACO ---------------------------------------------------------------
TACO_difficulties_list = ['EASY', 'MEDIUM', 'MEDIUM_HARD', 'HARD', 'VERY_HARD']
TACO_reader_cfg = dict(input_columns=['question', 'starter'], output_column='problem_id', train_split='test')

TACO_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='Please write a python program to address the following QUESTION. Your ANSWER should be in a code block format like this: ```python # Write your code here ```. \nQUESTION:\n{question} {starter}\nANSWER:\n'),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=1024),
)

TACO_eval_cfg = dict(evaluator=dict(type=TACOEvaluator), pred_role='BOT')

for difficulty in TACO_difficulties_list:
    compassbench_v1_1_code_datasets.append(
        dict(
            type=TACODataset,
            abbr='TACO-' + difficulty,
            path='data/compassbench_v1.1/code/BAAI-TACO',
            difficulty=difficulty,
            reader_cfg=TACO_reader_cfg,
            infer_cfg=TACO_infer_cfg,
            eval_cfg=TACO_eval_cfg,
        )
    )
