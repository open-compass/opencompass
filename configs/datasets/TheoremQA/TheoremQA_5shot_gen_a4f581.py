from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import HFDataset, TheoremQA_postprocess_v3, TheoremQAEvaluatorV3

TheoremQA_reader_cfg = dict(input_columns=["Question", "Answer_type"], output_column="Answer", train_split="test", test_split="test")

TheoremQA_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='You are supposed to provide a solution to a given problem.\n\n\nProblem:\nIn a 10 Gigabit Ethernet network, the average size of a frame is 1500 bytes. If a burst of noise lasting 1ms interrupts the network, how many frames are lost?'),
                dict(role='BOT', prompt='Solution:\nFirst, calculate the data rate in bytes/s:\n\n10 Gigabit/s * (1 Byte / 8 bits) = 1.25 * 10^9 Bytes/s\n\nNext, calculate the data loss in bytes due to the noise:\n\n1 ms * 1.25 * 10^9 Bytes/s = 1.25 * 10^6 Bytes\n\nFinally, divide the data loss by the average frame size to get the number of frames lost:\n\n1.25 * 10^6 Bytes / 1500 Bytes/frame ≈ 833.33 frames\nThe answer is 833.33'),
                dict(role='HUMAN', prompt='\nProblem:\nGiven x = 0.157, what is the value of x \\times \\frac{\\prod_{n=1}^\\infty (1 - \\frac{x^2}{n^2 \\pi^2})}{\\sin(x)}?'),
                dict(role='BOT', prompt="Solution:\nTo evaluate the expression $x \\times \\frac{\\prod_{n=1}^{\\infty} (1 - \\frac{x^2}{n^2 \\pi^2})}{\\sin(x)}$ given x = 0.157, we first recognize that the product in the numerator is related to the sine function through the Euler's reflection formula for the sine function, which can be expressed as:\n\n$$\\sin(x) = x \\prod_{n=1}^{\\infty} \\left(1 - \\frac{x^2}{n^2 \\pi^2}\\right)$$\n\nTherefore, the given expression simplifies to: $x \\times \\frac{\\sin(x)}{\\sin(x)}$\n\nBecause sin(x) in the numerator and denominator cancels out, the expression simplifies further to just x.\n\nSo, given x = 0.157, the value of the expression is 0.157. This result is derived from the properties of the sine function and does not require computational evaluation.\nThe answer is 0.157"),
                dict(role='HUMAN', prompt='\nProblem:\nConsider the basis C of \\mathbb{R}^2 consisting of vectors u_1 = [2, 4] and u_2 = [1, -1]. If y = [8, 12], find the C-coordinate vector of y.'),
                dict(role='BOT', prompt="Solution:\nThe goal is to express y as a linear comPbination of the basis vectors of C, i.e., $y = a\\cdot u_1 + b\\cdot u_2$, where a and b are the scalar coefficients that we want to find. These coefficients will form the C-coordinate vector of y, which we'll denote as $[a, b]_C$.\n\nGiven:\n- $u_1 = [2, 4]$,\n- $u_2 = [1, -1]$,\n- $y = [8, 12]$.\n\nWe need to solve the system of linear equations:\n2a + 1b = 8\n4a - 1b = 12\n\nLet's solve this system of equations to find a and b.\n\nThe solution to the system of equations is $a = \\frac{10}{3} and b = \\frac{4}{3}$. Therefore, the C-coordinate vector of y in the basis consisting of vectors u_1 = [2, 4] and u_2 = [1, -1] is $\\left[\\frac{10}{3}, \\frac{4}{3}\\right]_C$. \nLet's calculate the numerical value of $\\left[\x0crac{10}{3}, \x0crac{4}{3}\right]_C$ as [3.33, 1.33].\nThe answer is [3.33, 1.33]"),
                dict(role='HUMAN', prompt='\nProblem:\nOne can draw a simple, connected planar graph with 200 vertices and 397 edges. Is this statement Trur or False?'),
                dict(role='BOT', prompt="Solution:\nTo determine the answer, we can use Euler's formula for planar graphs, which states that for any finite, connected, planar graph, $V - E + F = 2$, where V is the number of vertices, E is the number of edges, and F is the number of faces.\n\nGiven the modified question, we have V = 200 vertices and E = 397 edges. We want to find if we can have a graph that satisfies these conditions, adhering to Euler's formula.\n\nFirst, let's rearrange Euler's formula to solve for F:  F = E - V + 2\n\nSubstituting the given values: F = 397 - 200 + 2,  F = 199\n\nThis means a graph with 200 vertices and 397 edges would have 199 faces. However, to determine the truth of this possibility, we should check if this graph doesn't violate any other planar graph constraints, particularly regarding the number of edges.\n\nFor a simple, connected planar graph, there's also a relationship between vertices, edges, and faces given by the inequality: $E \\leq 3V - 6$\n\nSubstituting V = 200 gives: $E \\leq 3*200 - 6 = 594$\n\nWith E = 397, the condition $E \\leq 594$ is satisfied, meaning it's theoretically possible in terms of the edge condition for a planar graph.\n\nTherefore, one can draw a simple, connected planar graph with 200 vertices and 397 edges, resulting in 199 faces, without violating the conditions for it to be planar according to both Euler's formula and the constraint on the maximum number of edges.\nThe answer is True"),
                dict(role='HUMAN', prompt='\nProblem:\nGiven a finite group G, and a collection of permutations H on a set. Then (a) there always exists H such that G is isomorphic to H; (b) for any H, G is isomorphic to H; (c) G can never be isomorphic to H; (d) none of the above. Which option is correct?'),
                dict(role='BOT', prompt="Solution:\nThis is based on Cayley's theorem, which states that every group G is isomorphic to a subgroup of the symmetric group acting on G. \nIn other words, for every finite group G, there exists a collection of permutations H (which in this context, can be thought of as the set of permutations representing the action of G on itself) such that G is isomorphic to H.\n\nTherefore, there always exists H such that G is isomorphic to H.\nThe answer is (a)"),
                dict(role='HUMAN', prompt='\nProblem:\n{Question}'),
                dict(role='BOT', prompt='Solution:\n{Answer}'),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=1024, stopping_criteria=["USER:", "ASSISTANT:",  "### Instruction:", "Response:", "<start_of_turn>", "[INST]", "Problem:"]),
)

TheoremQA_eval_cfg = dict(
    evaluator=dict(type=TheoremQAEvaluatorV3),
    pred_postprocessor=dict(type=TheoremQA_postprocess_v3)
)

TheoremQA_datasets = [
    dict(
        abbr="TheoremQA",
        type=HFDataset,
        path="TIGER-Lab/TheoremQA",
        reader_cfg=TheoremQA_reader_cfg,
        infer_cfg=TheoremQA_infer_cfg,
        eval_cfg=TheoremQA_eval_cfg,
    )
]
