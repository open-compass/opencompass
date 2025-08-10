from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import SanitizedMBPPDataset, MBPPEvaluator

sanitized_mbpp_reader_cfg = dict(input_columns=['text', 'test_list'], output_column='test_list_2')

sanitized_mbpp_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='''\
You are an expert Python programmer, and here is your task: Write a function to find the similar elements from the given two tuple lists. Your code should pass these tests:

assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)
assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)
assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)

[BEGIN]
 'def similar_elements(test_tup1, test_tup2):
  res = tuple(set(test_tup1) & set(test_tup2))
  return (res)'
[DONE]



You are an expert Python programmer, and here is your task: Write a python function to identify non-prime numbers. Your code should pass these tests:

assert is_not_prime(2) == False
assert is_not_prime(10) == True
assert is_not_prime(35) == True

[BEGIN]
 'import math
def is_not_prime(n):
    result = False
    for i in range(2,int(math.sqrt(n)) + 1):
        if n % i == 0:
            result = True
    return result'
[DONE]



You are an expert Python programmer, and here is your task: Write a function to find the largest integers from a given list of numbers using heap queue algorithm. Your code should pass these tests:

assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65]
assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75]
assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]

[BEGIN]
 'import heapq as hq
def heap_queue_largest(nums,n):
  largest_nums = hq.nlargest(n, nums)
  return largest_nums'
[DONE]



You are an expert Python programmer, and here is your task: {text} Your code should pass these tests:

{test_list}

[BEGIN]
'''
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512),
)

sanitized_mbpp_eval_cfg = dict(evaluator=dict(type=MBPPEvaluator), pred_role='BOT')

sanitized_mbpp_datasets = [
    dict(
        type=SanitizedMBPPDataset,
        abbr='sanitized_mbpp',
        path='./data/mbpp/sanitized-mbpp.jsonl',
        reader_cfg=sanitized_mbpp_reader_cfg,
        infer_cfg=sanitized_mbpp_infer_cfg,
        eval_cfg=sanitized_mbpp_eval_cfg,
    )
]
