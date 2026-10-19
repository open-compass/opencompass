from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import MBPPDatasetV2, MBPPPassKEvaluator

mbpp_reader_cfg = dict(input_columns=['text', 'test_list'], output_column='test_column')

mbpp_infer_cfg = dict(
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

mbpp_eval_cfg = dict(evaluator=dict(type=MBPPPassKEvaluator), pred_role='BOT')

mbpp_datasets = [
    dict(
        type=MBPPDatasetV2,
        abbr='mbpp_passk',
        path='./data/mbpp/mbpp.jsonl',
        reader_cfg=mbpp_reader_cfg,
        infer_cfg=mbpp_infer_cfg,
        eval_cfg=mbpp_eval_cfg,
    )
]
