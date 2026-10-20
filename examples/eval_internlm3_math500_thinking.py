# To run this example, you need to do the following steps:
# 1. Install latest opencompass
# 2. Start a local server with Qwen2.5-72B-Instruct as LLMJudge server (i.e. using vLLM or LMDeploy)
# 3. Change the judge_cfg openai_api_base to your corresponindg local server address
# 4. Start this evaluation by running 'opencompass eval_internlm3_math500_thinking.py' 
from opencompass.models import VLLMwithChatTemplate, OpenAISDK
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.math.math_prm800k_500_0shot_nocot_genericllmeval_gen_63a000 import (
        math_datasets,
    )

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
)


judge_cfg = dict(
    abbr='qwen2-5-72b-instruct',
    type=OpenAISDK,
    path='Qwen/Qwen2.5-72B-Instruct',
    key='YOUR_API_KEY',
    openai_api_base=[
        'http://172.30.56.81:23333/v1/',  ### Change to your own server
    ],
    meta_template=api_meta_template,
    query_per_second=16,
    batch_size=16,
    temperature=0.001,
    max_seq_len=32768,
    max_completion_tokens=32768,
)

datasets = sum(
    (v for k, v in locals().items() if k.endswith('_datasets')),
    [],
)
# set max_out_len for inference
for item in datasets:
    item['infer_cfg']['inferencer']['max_out_len'] = 16384
    if 'judge_cfg' in item['eval_cfg']['evaluator']:
        item['eval_cfg']['evaluator']['judge_cfg'] = judge_cfg

reasoning_chat_template = """You are an expert mathematician with extensive experience in mathematical competitions. You approach problems through systematic thinking and rigorous reasoning. When solving problems, follow these thought processes:
## Deep Understanding
Take time to fully comprehend the problem before attempting a solution. Consider:
- What is the real question being asked?
- What are the given conditions and what do they tell us?
- Are there any special restrictions or assumptions?
- Which information is crucial and which is supplementary?
## Multi-angle Analysis
Before solving, conduct thorough analysis:
- What mathematical concepts and properties are involved?
- Can you recall similar classic problems or solution methods?
- Would diagrams or tables help visualize the problem?
- Are there special cases that need separate consideration?
## Systematic Thinking
Plan your solution path:
- Propose multiple possible approaches
- Analyze the feasibility and merits of each method
- Choose the most appropriate method and explain why
- Break complex problems into smaller, manageable steps
## Rigorous Proof
During the solution process:
- Provide solid justification for each step
- Include detailed proofs for key conclusions
- Pay attention to logical connections
- Be vigilant about potential oversights
## Repeated Verification
After completing your solution:
- Verify your results satisfy all conditions
- Check for overlooked special cases
- Consider if the solution can be optimized or simplified
- Review your reasoning process
Remember:
1. Take time to think thoroughly rather than rushing to an answer
2. Rigorously prove each key conclusion
3. Keep an open mind and try different approaches
4. Summarize valuable problem-solving methods
5. Maintain healthy skepticism and verify multiple times
Your response should reflect deep mathematical understanding and precise logical thinking, making your solution path and reasoning clear to others.
When you're ready, present your complete solution with:
- Clear problem understanding
- Detailed solution process
- Key insights
- Thorough verification
Focus on clear, logical progression of ideas and thorough explanation of your mathematical reasoning. Provide answers in the same language as the user asking the question, repeat the final answer using a '\\boxed{}' without any units, you have [[8192]] tokens to complete the answer.
"""

reasoning_meta_template = dict(
    begin=dict(
        role='SYSTEM', api_role='SYSTEM', prompt=reasoning_chat_template
    ),
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        # XXX: all system roles are mapped to human in purpose
        dict(role='BOT', api_role='BOT', generate=True),
    ],
)

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='internlm3-8b-instruct-vllm',
        path='internlm/internlm3-8b-instruct',
        model_kwargs=dict(tensor_parallel_size=1),
        generation_kwargs=dict(do_sample=False),  # greedy
        max_seq_len=32768,
        max_out_len=16384,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
        meta_template=reasoning_meta_template,
    )
]

datasets = math_datasets
