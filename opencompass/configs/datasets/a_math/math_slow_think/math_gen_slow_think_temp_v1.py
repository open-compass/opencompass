from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import MATHDataset, MATHEvaluator, math_postprocess_v2

math_reader_cfg = dict(input_columns=['problem'], output_column='solution')
stop_words = ["</s>", "<|im_end|>", "<|endoftext|>","<|end_of_text|>","\nught","\nughtuser","\nQuestion","\nnoteq","\nosterone","nosteroneuser"]

prompt = """
You are given a general approach to solving mathematical problems. Please refer to the reasoning approach demonstrated below and solve the question by yourself:

1. **Understand the problem**: Ensure you correctly read the problem statement. Identify known values, unknowns, and any implied conditions. Pay attention to keywords, units, or graphical information in the problem.

2. **Visualize if necessary**: For geometry or application problems, drawing diagrams or working with examples might help. If the problem is abstract, visualization can aid understanding. Also, identify the problem type, whether it’s an equation-solving problem, proof, optimization, etc. Different types of problems require different methods.

3. **Make a plan**: Recall relevant theorems, formulas, or methods for solving similar problems. For instance, solving a quadratic equation might involve factoring or using the quadratic formula, while geometric proofs might require similarity of triangles or the Pythagorean theorem. If you forget a related concept, consult textbooks or references. In an exam, rely on your memory.

4. **Execute the plan step by step**: Carefully work through each step, paying attention to intermediate calculations to avoid errors, especially with signs or small mistakes. Check each step to ensure correctness.

5. **Verify the result**: Substitute the solution back into the original problem to confirm it satisfies all conditions. Alternatively, try solving the problem using a different method to verify consistency. Consider whether multiple solutions are possible (e.g., quadratic equations often have two solutions) and check if both satisfy the problem.

6. **If you encounter difficulties**: Return to earlier steps and analyze again. Try different methods if necessary. For instance, if algebraic methods don’t work, try a geometric approach, or break down the problem into smaller parts and solve them step by step.

7. **Reflect and summarize**: After solving the problem, reflect on the techniques used. Think about any potential optimizations or alternative solutions. Consider the underlying mathematical concepts, such as transformation methods or combining algebra with geometry. This reflection helps improve problem-solving skills.

Now, solve the following question:

{problem}

Finally, provide your answer in the following format: 'So the answer is $\\boxed{}$'.
"""

math_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt=prompt),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=4096,stopping_criteria=stop_words),
)

# postprocess v2
math_eval_cfg = dict(
    evaluator=dict(type=MATHEvaluator, version='v2'), pred_postprocessor=dict(type=math_postprocess_v2),
)

math_datasets = [
    dict(
        type=MATHDataset,
        abbr='math_slow_think_temp_v1',
        path='opencompass/math',
        reader_cfg=math_reader_cfg,
        infer_cfg=math_infer_cfg,
        eval_cfg=math_eval_cfg,
    )
]
