from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator

gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

stop_words = ["</s>", "<|im_end|>", "<|endoftext|>","\n\nQuestion:","<|end_of_text|>"," Question","\n[Question]"]


prompt = """
You may encounter algebraic problems, geometric problems, or more complex calculus problems, so the approach needs to be flexible enough. Below is a general problem-solving template that you should follow:

First, the key step in solving a problem is to ensure that you have understood the problem correctly. This includes clarifying the known conditions and unknowns, as well as identifying any implicit conditions. Pay attention to keywords, units, or graphical information in the problem.

Next, you may need to draw diagrams or give examples to help with understanding, especially in geometry or application problems. If the problem is abstract, visualization can be helpful. You should also identify the type of problem, such as equation solving, proof, or optimization, as different types may require different methods.

After that, you need to formulate a solution plan. Here, you may need to recall relevant theorems, formulas, or methods for solving similar problems. For example, solving a quadratic equation may require factoring or using the quadratic formula, while a geometric proof may need similar triangles or the Pythagorean theorem. If you can’t recall the relevant knowledge, you may need to consult reference materials, but in an exam situation, you must rely on memory.

Then, execute your plan step by step, ensuring each intermediate step is correct. It’s important to be cautious of common errors, such as sign mistakes or calculation oversights, so you should double-check every step.

Verification of the solution is also essential. You can substitute the solution back into the original problem to check if it satisfies all conditions. Alternatively, try solving the problem with a different method to confirm consistency. Consider whether there might be multiple solutions, such as with quadratic equations, and ensure both solutions make sense.

If you encounter difficulties, you may need to revisit earlier steps and reanalyze the problem. Alternatively, try different methods. For instance, if an algebraic approach doesn’t work, try a geometric method, or break the problem into smaller parts and solve them one by one before synthesizing the final solution.

Finally, reflect on the entire process. Think about the methods you used and whether a more efficient solution might exist. Consider the mathematical principles behind the problem, such as transformation techniques or the relationship between algebra and geometry. This reflection is vital for improving problem-solving skills.

When facing difficult problems, mental strategies are important. Stay calm and, if necessary, revisit the problem's implicit information to uncover insights. It’s also crucial to maintain logical rigor, particularly in proof-based problems where every step must be justified.

Remember that different mathematical domains may require distinct strategies. For example, probability problems might involve listing all possible events, while calculus problems may need differentiation or integration techniques. However, the general steps outlined here can be adapted to any type of problem, with specific methods adjusted accordingly.

In summary, the general steps are: Understand the problem, formulate a plan, execute the plan, and verify and reflect. Each of these steps requires careful consideration.

Please refer to the reasoning approach demonstrated above, solve the following question by yourself:
{question}

Finally, provide your answer in the following format: 'So the answer is $\\boxed{}$'.
"""


gsm8k_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt=prompt),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=4096,stopping_criteria=stop_words)
)

gsm8k_eval_cfg = dict(
    evaluator=dict(type=Gsm8kEvaluator),
    pred_postprocessor=dict(type=gsm8k_postprocess),
    dataset_postprocessor=dict(type=gsm8k_dataset_postprocess),
)

gsm8k_datasets = [
    dict(
        abbr='gsm8k_slow_think_temp_v3',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg,
    )
]
