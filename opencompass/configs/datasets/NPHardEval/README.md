# NPHardEval
## Introduction
The following introduction comes from the abstract in [NPHardEval: Dynamic Benchmark on Reasoning Ability of Large Language Models via Complexity Classes](https://arxiv.org/abs/2312.14890)

```
This benchmark is designed to evaluate the reasoning abilities of LLMs across a broad spectrum of 900 algorithmic questions, extending up to the NP-Hard complexity class. These questions are meticulously chosen to represent a wide range of complexity class below the NP-hard complexity class, offering a rigorous measure of the reasoning ability of LLMs.
```

## Official link

### Paper

[NPHardEval: Dynamic Benchmark on Reasoning Ability of Large Language Models via Complexity Classes](https://arxiv.org/abs/2312.14890)

### Repository

[NPHardEval](https://github.com/casmlab/NPHardEval)

## Examples
Input example I:
```
The Graph Coloring Problem (Decision Version, GCP-D) involves determining if it is possible to color the vertices of a graph using a given number of colors, ensuring no two adjacent vertices have the same color.
Find out if the vertices of a graph with 6 vertices can be colored using only 3 colors, such that no adjacent vertices share the same color.
Provide a yes or no answer, along with a concise explanation of your reasoning. Keep your explanation focused and brief.
Enclose your output in <root></root> tags. Include your reasoning in <reasoning></reasoning> tags and the final yes/no answer in <final_answer></final_answer> tags, like <final_answer>{'Feasible': 'YES_OR_NO'}</final_answer>.

 The graph is below:
Vertex 2 is connected to vertex 4.
Vertex 2 is connected to vertex 5.
Vertex 3 is connected to vertex 5.
Vertex 4 is connected to vertex 5.
Vertex 5 is connected to vertex 6.
```

Output example I (from GPT-4):
```
<root>
<final_answer>{'Feasible': 'YES'}</final_answer>
<reasoning>Yes, it is possible to color the vertices of the given graph with 6 vertices using only 3 colors, such that no adjacent vertices share the same color. The graph does not contain any triangles or cycles of length greater than 3, which means it is a planar graph. By the Four Color Theorem, any planar graph can be colored using at most 4 colors. Since we have only 3 colors available, it is feasible to color the vertices without any adjacent vertices sharing the same color.</reasoning>
</root>
```

Input example II:
```
The Traveling Salesman Problem (Decision Version, TSP-D) focuses on determining if a salesman can complete a route, visiting each city at least once, with the total travel distance being less than a specified value.
Check if it's possible for a salesman to visit each of the 10 cities at least once and return to the starting city with the total distance less than 3481.5. The distances between each pair of cities are given.
Provide a yes or no answer, with a succinct explanation of your decision process. Focus on clarity and brevity in your response.
Enclose your output in <root></root> tags. Present your reasoning in <reasoning></reasoning> tags and the final yes/no answer in <final_answer></final_answer> tags, like <final_answer>{'Feasible': 'YES_OR_NO'}</final_answer>.
 The distances between cities are below:
The distance between City 0 and City 1 is 58.0.
The distance between City 0 and City 2 is 171.0.
The distance between City 0 and City 3 is 122.0.
The distance between City 0 and City 4 is 109.0.
The distance between City 0 and City 5 is 93.0.
The distance between City 0 and City 6 is 106.0.
The distance between City 0 and City 7 is 52.0.
The distance between City 0 and City 8 is 115.0.
The distance between City 0 and City 9 is 148.0.
The distance between City 1 and City 2 is 145.0.
The distance between City 1 and City 3 is 71.0.
The distance between City 1 and City 4 is 114.0.
The distance between City 1 and City 5 is 69.0.
The distance between City 1 and City 6 is 163.0.
The distance between City 1 and City 7 is 132.0.
The distance between City 1 and City 8 is 183.0.
The distance between City 1 and City 9 is 115.0.
The distance between City 2 and City 3 is 107.0.
The distance between City 2 and City 4 is 163.0.
The distance between City 2 and City 5 is 63.0.
The distance between City 2 and City 6 is 91.0.
The distance between City 2 and City 7 is 108.0.
The distance between City 2 and City 8 is 109.0.
The distance between City 2 and City 9 is 77.0.
The distance between City 3 and City 4 is 119.0.
The distance between City 3 and City 5 is 131.0.
The distance between City 3 and City 6 is 79.0.
The distance between City 3 and City 7 is 159.0.
The distance between City 3 and City 8 is 27.0.
The distance between City 3 and City 9 is 64.0.
The distance between City 4 and City 5 is 85.0.
The distance between City 4 and City 6 is 110.0.
The distance between City 4 and City 7 is 71.0.
The distance between City 4 and City 8 is 61.0.
The distance between City 4 and City 9 is 61.0.
The distance between City 5 and City 6 is 73.0.
The distance between City 5 and City 7 is 132.0.
The distance between City 5 and City 8 is 82.0.
The distance between City 5 and City 9 is 96.0.
The distance between City 6 and City 7 is 156.0.
The distance between City 6 and City 8 is 103.0.
The distance between City 6 and City 9 is 69.0.
The distance between City 7 and City 8 is 108.0.
The distance between City 7 and City 9 is 97.0.
The distance between City 8 and City 9 is 75.0.
```

Output example II (from GPT-4):
```
<root> <reasoning> The Traveling Salesman Problem is a well-known problem in the field of computer science and operations research. It is a combinatorial optimization problem, which means it involves finding the best solution from a finite set of possible solutions. The problem is NP-hard, which means there is no known algorithm that can solve it in polynomial time. Therefore, to solve this problem, we would need to use an algorithm that can find the optimal solution, such as a brute force algorithm, a genetic algorithm, or a simulated annealing algorithm. However, these algorithms can be computationally expensive, especially for large problem sizes. In this case, we have 10 cities, which means there are 10! = 3,628,800 possible routes. Therefore, without the use of a computer, it would be impractical to manually check all possible routes to find the one with the total distance less than 3481.5. </reasoning> <final_answer>{'Feasible': 'UNKNOWN'}</final_answer> </root>
```

## Evaluation results

```
dataset    version    metric             mode      internlm2-chat-7b-hf
---------  ---------  -----------------  ------  ----------------------
hard_GCP   144a59     Weighted Accuracy  gen                       1.64
hard_TSP   144a59     Weighted Accuracy  gen                       0
hard_MSP   144a59     Weighted Accuracy  gen                       0
cmp_GCP_D  144a59     Weighted Accuracy  gen                      43.82
cmp_TSP_D  144a59     Weighted Accuracy  gen                      40.18
cmp_KSP    144a59     Weighted Accuracy  gen                       0
p_BSP      144a59     Weighted Accuracy  gen                      40.36
p_EDP      144a59     Weighted Accuracy  gen                       0
p_SPP      144a59     Weighted Accuracy  gen                       0
```

## Reference
```
@article{fan2023nphardeval,
  title={NPHardEval: Dynamic Benchmark on Reasoning Ability of Large Language Models via Complexity Classes},
  author={Fan, Lizhou and Hua, Wenyue and Li, Lingyao and Ling, Haoyang and Zhang, Yongfeng and Hemphill, Libby},
  journal={arXiv preprint arXiv:2312.14890},
  year={2023}
}
```
