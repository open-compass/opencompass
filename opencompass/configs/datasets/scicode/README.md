# SciCode: A Research Coding Benchmark Curated by Scientists

## Introduction
SciCode is a challenging benchmark designed to evaluate the capabilities of language models (LMs) in generating code for solving realistic scientific research problems. It has a diverse coverage of 16 subdomains from 6 domains: Physics, Math, Material Science, Biology, and Chemistry. Unlike previous benchmarks that consist of exam-like question-answer pairs, SciCode is converted from real research problems. SciCode problems naturally factorize into multiple subproblems, each involving knowledge recall, reasoning, and code synthesis. In total, SciCode contains 338 subproblems decomposed from 80 challenging main problems, and it offers optional descriptions specifying useful scientific background information and scientist-annotated gold-standard solutions and test cases for evaluation. Claude3.5-Sonnet, the best-performing model among those tested, can solve only 4.6% of the problems in the most realistic setting. Broadly, SciCode demonstrates a realistic and scientists' everyday workflow of identifying critical science concepts and facts and then transforming them into computation and simulation code. We believe SciCode not only helps demonstrate contemporary LLMs' progress towards helpful assistant for scientists but also helps shed light on future building and evaluation of scientific AI. For more detailed information, please refer to https://scicode-bench.github.io/.

## How to Use
By modifying the with_bg parameter in the configuration file, you can support setup for w/ background evaluation.

```bash
python run.py --datasets scicode_gen --hf-num-gpus 1 --hf-type chat --hf-path meta-llama/Meta-Llama-3-8B-Instruct --debug --model-kwargs device_map='auto' trust_remote_code=True --batch-size 1
```

## Reference Performance
| Model                     | Condition    | Subproblem Accuracy | Main Problem Accuracy |
|---------------------------|--------------|---------------------|-----------------------|
| Llama-3-70B-Instruct      | w/o Background  | 21.53%              | 4.62%                  |
| Llama-3-70B-Instruct      | w/ Background   | 24.31%              | 7.69%                  |
| Qwen2-72B-Instruct        | w/o Background  | 16.67%              | 1.54%                  |
| Qwen2-72B-Instruct        | w/ Background   | 19.79%              | 1.54%                  |

## Citation
```
@misc{tian2024scicode,
    title={SciCode: A Research Coding Benchmark Curated by Scientists},
    author={Minyang Tian and Luyu Gao and Shizhuo Dylan Zhang and Xinan Chen and Cunwei Fan and Xuefei Guo and Roland Haas and Pan Ji and Kittithat Krongchon and Yao Li and Shengyan Liu and Di Luo and Yutao Ma and Hao Tong and Kha Trinh and Chenyu Tian and Zihan Wang and Bohao Wu and Yanyu Xiong and Shengzhu Yin and Minhui Zhu and Kilian Lieret and Yanxin Lu and Genglin Liu and Yufeng Du and Tianhua Tao and Ofir Press and Jamie Callan and Eliu Huerta and Hao Peng},
    year={2024},
    eprint={2407.13168},
    archivePrefix={arXiv},
    primaryClass={cs.AI}
}
```
