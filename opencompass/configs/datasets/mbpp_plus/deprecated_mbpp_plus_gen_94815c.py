from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import MBPPEvaluator, MBPPPlusDataset

mbpp_plus_reader_cfg = dict(
    input_columns=['text', 'test_list'], output_column='task_id')

mbpp_plus_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt=
                    'You are an expert Python programmer, and here is your task: Write a function to find the shared elements from the given two lists. Your code should pass these tests:\n\n assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)\n assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4) \n assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14) \n'
                ),
                dict(
                    role='BOT',
                    prompt=
                    "[BEGIN]\n 'def similar_elements(test_tup1, test_tup2):\n  return tuple(set(test_tup1) & set(test_tup2))' \n[DONE] \n\n "
                ),
                dict(
                    role='HUMAN',
                    prompt=
                    'You are an expert Python programmer, and here is your task: Write a python function to identify non-prime numbers. Your code should pass these tests:\n\n assert is_not_prime(2) == False \n assert is_not_prime(10) == True \n assert is_not_prime(35) == True \n'
                ),
                dict(
                    role='BOT',
                    prompt=
                    "[BEGIN]\n 'import math\ndef is_not_prime(n):\n    if n == 1:\n        return True\n    for i in range(2, int(math.sqrt(n))+1):\n        if n % i == 0:\n            return True\n    return False' \n[DONE] \n\n "
                ),
                dict(
                    role='HUMAN',
                    prompt=
                    'You are an expert Python programmer, and here is your task: Write a function to find the n largest integers from a given list of numbers, returned in descending order. Your code should pass these tests:\n\n assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] \n assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] \n assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35] \n'
                ),
                dict(
                    role='BOT',
                    prompt=
                    "[BEGIN]\n 'import heapq as hq\ndef heap_queue_largest(nums: list,n: int) -> list:\n  largest_nums = hq.nlargest(n, nums)\n  return largest_nums' \n[DONE] \n\n "
                ),
                dict(
                    role='HUMAN',
                    prompt=
                    'You are an expert Python programmer, and here is your task: {text} Your code should pass these tests:\n\n {test_list}  \n'
                ),
                dict(role='BOT', prompt='[BEGIN]\n'),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512))

mbpp_plus_eval_cfg = dict(evaluator=dict(type=MBPPEvaluator, metric='MBPPPlus'), pred_role='BOT')

mbpp_plus_datasets = [
    dict(
        type=MBPPPlusDataset,
        abbr='mbpp_plus',
        path='./data/mbpp_plus/mbpp_plus.jsonl',
        reader_cfg=mbpp_plus_reader_cfg,
        infer_cfg=mbpp_plus_infer_cfg,
        eval_cfg=mbpp_plus_eval_cfg)
]
