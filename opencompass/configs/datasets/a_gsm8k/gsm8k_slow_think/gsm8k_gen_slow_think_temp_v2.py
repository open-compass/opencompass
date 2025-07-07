from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator

gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

stop_words = ["</s>", "<|im_end|>", "<|endoftext|>","\n\nQuestion:","<|end_of_text|>"," Question","\n[Question]"]


prompt = """
You are given a structured and systematic approach to solving mathematical problems. Please refer to the reasoning approach demonstrated below and solve the following question by yourself:

1. **Fully understand the problem**:
   - **Clarify given conditions**: Identify all provided values, relationships, formulas, and graphical information.
   - **Confirm the objective**: Clearly understand what the problem is asking for (e.g., solving for an unknown, proving a conclusion, constructing an example, etc.).
   - **Identify hidden information**: Check for implicit conditions such as units, domain restrictions, symmetries (e.g., “positive integer solutions” or “continuous function”).
   - **Use examples for clarification**: Try substituting specific values for abstract problems, or draw diagrams (e.g., for geometry or function graphs).

2. **Analyze the type of problem**:
   - **Classify the problem**: Determine whether it belongs to algebra, geometry, combinatorics, calculus, or other subfields.
   - **Recall similar problem types**: Think of any previously encountered problems with similar structures (e.g., “chickens and rabbits” problem and linear equations).
   - **Identify key knowledge points**: Match the problem with relevant theorems or formulas (e.g., Pythagorean theorem, mean value theorem, permutation and combination formulas).

3. **Develop a problem-solving strategy**:
   - **Break down the problem**: Divide complex problems into simpler subproblems (e.g., solving a differential equation step by step).
   - **Choose tools**:
     - **Algebra**: Equation manipulation, factorization, inequality scaling.
     - **Geometry**: Auxiliary lines, coordinate transformations, similarity/congruence of triangles.
     - **Analysis**: Techniques in limits, derivatives, and integrals.
     - **Combinatorics**: Casework, recursion, inclusion-exclusion principle.
   - **Think in reverse**: Consider working backward from the goal (e.g., for a proof, assume the conclusion and derive the necessary conditions).

4. **Execute the plan and record the process**:
   - **Step-by-step derivation**: Write out each step clearly to avoid skipping steps (especially where mistakes are common, such as sign errors).
   - **Mark uncertain steps**: Flag any steps you are unsure about for further validation later.
   - **Simplify computations**: Use symmetries, special values, or variable substitutions to reduce complexity (e.g., setting \( t = x + \frac{1}{x} \)).

5. **Verification and correction**:
   - **Substitute and check**: Plug the result back into the original problem to ensure it satisfies all conditions (e.g., check roots in an equation).
   - **Cross-check using different methods**: Solve the same problem with different approaches (e.g., using both analytic geometry and pure geometry for a geometry problem).
   - **Test boundary conditions**: Check extreme cases (e.g., try \( x = 0 \) or \( x \to \infty \)).
   - **Look for logical flaws**: Ensure each step is reversible or satisfies necessary theorem conditions (e.g., check that L’Hopital’s Rule is applied correctly to an indeterminate form).

6. **Handling stuck situations**:
   - **Restate the problem**: Try rephrasing the problem in a different language (e.g., converting a geometry problem into an algebraic equation).
   - **Start from special cases**: Look for patterns by considering special cases (e.g., explore the pattern for \( n = 1, 2, 3 \)).
   - **Use analogies from other fields**: Refer to techniques from other areas (e.g., apply probability theory to combinatorics).
   - **Take a break**: Step away and return later with a fresh perspective (this is the “incubation effect” in psychology).

7. **Summarize and extend**:
   - **Record key techniques**: Note down new methods or insights learned (e.g., “use of auxiliary functions to prove inequalities”).
   - **Generalize the problem**: Think about whether the problem can be extended (e.g., generalizing a 2D problem to n-dimensions).
   - **Connect to broader knowledge**: Relate this problem to other topics you have studied (e.g., recognizing this as a special case of a Taylor series).

Now, solve the following question:

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
        abbr='gsm8k_slow_think_temp_v2',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg,
    )
]
