# ArenaHard
## Introduction
The following introduction comes from the official repo:
Arena-Hard is an evaluation tool for instruction-tuned LLMs. It contains 500 challenging user queries, which prompt GPT-4-Turbo as judge to compare the models' responses against a baseline model (default: GPT-4-0314).

## Official link
https://github.com/lm-sys/arena-hard

### Paper
https://lmsys.org/blog/2024-04-19-arena-hard/

## Examples
Input example I:
```
Use ABC notation to write a melody in the style of a folk tune.
```

Output example I (from GPT-4):
```
X:1\nT:Untitled Folk Tune\nM:4/4\nL:1/8\nK:G\n|:G2A2|B2A2|G2E2|D4|E2F2|G2F2|E2C2|B,4|\nA2B2|c2B2|A2F2|E4|D2E2|F2E2|D2B,2|C4:|
```


## Evaluation results

```
LLaMa3-8b-instruct: 20.6 (Official Results)
LLaMa3-8b-instruct: 21.9 (Opencompass Results)
```

## Reference
```
@misc{arenahard2024,
    title = {From Live Data to High-Quality Benchmarks: The Arena-Hard Pipeline},
    url = {https://lmsys.org/blog/2024-04-19-arena-hard/},
    author = {Tianle Li*, Wei-Lin Chiang*, Evan Frick, Lisa Dunlap, Banghua Zhu, Joseph E. Gonzalez, Ion Stoica},
    month = {April},
    year = {2024}
}
```
