from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import SanitizedMBPPDataset, MBPPEvaluator

sanitized_mbpp_reader_cfg = dict(input_columns=['text', 'test_list'], output_column='test_list_2')

sanitized_mbpp_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='You are an expert Python programmer, and here is your task:\nWrite a function to find the similar elements from the given two tuple lists.\nYour code should pass these tests:\n\nassert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)\nassert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)\nassert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)\n',),
                dict(role='BOT', prompt='```python\ndef similar_elements(test_tup1, test_tup2):\n    res = tuple(set(test_tup1) & set(test_tup2))\n    return (res)```',),

                dict(role='HUMAN', prompt='You are an expert Python programmer, and here is your task:\nWrite a python function to identify non-prime numbers.\nYour code should pass these tests:\n\nassert is_not_prime(2) == False\nassert is_not_prime(10) == True\nassert is_not_prime(35) == True\n',),
                dict(role='BOT', prompt='```python\nimport math\ndef is_not_prime(n):\n    result = False\n    for i in range(2,int(math.sqrt(n)) + 1):\n        if n %% i == 0:\n            result = True\n    return result```',),

                dict(role='HUMAN', prompt='You are an expert Python programmer, and here is your task:\nWrite a function to find the largest integers from a given list of numbers using heap queue algorithm.\nYour code should pass these tests:\n\nassert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65]\nassert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75]\nassert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]\n',),
                dict(role='BOT', prompt='```python\nimport heapq as hq\ndef heap_queue_largest(nums,n):\n    largest_nums = hq.nlargest(n, nums)\n    return largest_nums```',),

                dict(role='HUMAN', prompt='You are an expert Python programmer, and here is your task:\n{text}\nYour code should pass these tests:\n\n{test_list}\n',),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512),
)

sanitized_mbpp_eval_cfg = dict(evaluator=dict(type=MBPPEvaluator), pred_role='BOT')

sanitized_mbpp_datasets = [
    dict(
        type=SanitizedMBPPDataset,
        abbr='sanitized_mbpp',
        path='opencompass/sanitized_mbpp',
        reader_cfg=sanitized_mbpp_reader_cfg,
        infer_cfg=sanitized_mbpp_infer_cfg,
        eval_cfg=sanitized_mbpp_eval_cfg,
    )
]
