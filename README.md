# CFinBench: A Comprehensive Chinese Financial Benchmark for Large Language Models

<div style="text-align: center;">
<img src="figs/intro.png" width="900" alt=""/>
  <br />
  <br />  </div>

[![license](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://github.com/yanbinwei/CFinBench-Eval/blob/main/LICENSE)

* [Introduction](#Introduction)
* [Installation](#Installation) 
* [Data Preparation](#data-preparation)
* [Evaluation](#evaluation)
* [Benchmark Results](#benchmark-results)
## Introduction
Welcome to **CFinBench**!

[[Project Page](https://cfinbench.github.io/)]

> Large language models (LLMs) have achieved remarkable performance on various NLP tasks, yet their potential in more challenging and domain-specific task, such as finance, has not been fully explored. In this paper, we present CFinBench: a meticulously crafted, the most comprehensive evaluation benchmark to date, for assessing the financial knowledge of LLMs under Chinese context. In practice, to better align with the career trajectory of Chinese financial practitioners, we build a systematic evaluation from 4 first-level categories: (1) Financial Subject: whether LLMs can memorize the necessary basic knowledge of financial subjects, such as economics, statistics and auditing. (2) Financial Qualification: whether LLMs can obtain the needed financial qualified certifications, such as certified public accountant, securities qualification and banking qualification. (3) Financial Practice: whether LLMs can fulfill the practical financial jobs, such as tax consultant, junior accountant and securities analyst. (4) Financial Law: whether LLMs can meet the requirement of financial laws and regulations, such as tax law, insurance law and economic law. CFinBench comprises 99,100 questions spanning 43 second-level categories with 3 question types: single-choice, multiple-choice and judgment. We conduct extensive experiments of 50 representative LLMs with various model size on CFinBench. The results show that GPT4 and some Chinese-oriented models lead the benchmark, with the highest average accuracy being 60.16%, highlighting the challenge presented by CFinBench. 

## Installation

This project is developed based on the OpenCompass evaluation framework. For detailed installation steps, please refer to [OpenCompass](https://github.com/open-compass/opencompass/tree/main).

```
conda create --name opencompass python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate opencompass
git clone https://github.com/open-compass/opencompass opencompass
cd opencompass
pip install -e .
```

## Data Preparation

To evaluate the CFinBench benchmark, first you need to prepare the CFinBench datasets. You can download the compressed dataset archive from [Project Page](https://cfinbench.github.io/) and execute the following commands.

```
# Download CFinBench to data/ folder and unzip
cd data
unzip CFinBench
```

After extracting the CFinBench. The file structure of the directory should be shown as follows. CFinBench is divided into three splits: dev, test, and val. The dev subset is used for the few-shot setting, the test subset has currently only released the questions, and the val subset has released both the questions and answers. You can perform evaluation on the val subset.

```
├── /data/CFinBench/
│  ├── /dev/
│  │  ├── /judgment/
│  │  │  ├── /1-1.jsonl
│  │  │  ├── /1-2.jsonl
│  │  │  └── ...
│  │  ├── /multi_choice/
│  │  │  ├── /1-1.jsonl
│  │  │  └── ...
│  │  └── /single_choice/
│  │  │  └── ...
│  ├── /test/
│  │  ├── /judgment/
│  │  │  └── ...
│  │  ├── /multi_choice/
│  │  │  └── ...
│  │  └── /single_choice/
│  │  │  └── ...
│  ├── /val/
│  │  ├── /judgment/
│  │  │  └── ...
│  │  ├── /multi_choice/
│  │  │  └── ...
│  │  └── /single_choice/
│  │  │  └── ...
```
The following table maps the file ID to its corresponding first-level and second-level category. Please refer to our paper for more detailed statistics of CFinBench.

<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky">First-Level Categories</th>
    <th class="tg-0pky">ID</th>
    <th class="tg-0pky">Second-Level Categories</th>
<th class="tg-0pky">First-Level Categories</th>
    <th class="tg-0pky">ID</th>
    <th class="tg-0pky">Second-Level Categories</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-lboi" rowspan="11"><b>Subject</b></td>
    <td class="tg-qdov">1-1.jsonl</td>
    <td class="tg-qdov">Political Economics</td>
    <td class="tg-lboi" rowspan="11"><b>Qualification</b></td>
    <td class="tg-qdov">2-1.jsonl</td>
    <td class="tg-qdov">Tax Practitioner Qualification</td>
  </tr>
  <tr>
    <td class="tg-0pky">1-2.jsonl</td>
    <td class="tg-qdov">Western Economics</td>
    <td class="tg-0pky">2-2.jsonl</td>
    <td class="tg-qdov">Futures Practitioner Qualification</td>
  </tr>
  <tr>
    <td class="tg-0pky">1-3.jsonl</td>
    <td class="tg-qdov">Microeconomics</td>
    <td class="tg-0pky">2-3.jsonl</td>
    <td class="tg-qdov">Fund Practitioner Qualification</td>
  </tr>
  <tr>
    <td class="tg-0pky">1-4.jsonl</td>
    <td class="tg-qdov">Macroeconomics</td>
    <td class="tg-0pky">2-4.jsonl</td>
    <td class="tg-qdov">Real Estate Practitioner Qualification</td>
  </tr>
  <tr>
    <td class="tg-0pky">1-5.jsonl</td>
    <td class="tg-qdov">Industrial Economics</td>
    <td class="tg-0pky">2-5.jsonl</td>
    <td class="tg-qdov">Insurance Practitioner Qualification</td>
  </tr>
  <tr>
    <td class="tg-0pky">1-6.jsonl</td>
    <td class="tg-qdov">Public Finance</td>
    <td class="tg-0pky">2-6.jsonl</td>
    <td class="tg-qdov">Securities Practitioner Qualification</td>
  </tr>
  <tr>
    <td class="tg-0pky">1-7.jsonl</td>
    <td class="tg-qdov">International Trade</td>
    <td class="tg-0pky">2-7.jsonl</td>
    <td class="tg-qdov">Banking Practitioner Qualification</td>
  </tr>
  <tr>
    <td class="tg-0pky">1-8.jsonl</td>
    <td class="tg-qdov">Statistics</td>
    <td class="tg-0pky">2-8.jsonl</td>
    <td class="tg-qdov">Certified Public Accountant (CPA)</td>
  </tr>
  <tr>
    <td class="tg-0pky">1-9.jsonl</td>
    <td class="tg-qdov">Auditing</td>
    <td class="tg-0pky"> </td>
    <td class="tg-qdov"> </td>
  </tr>
  <tr>
    <td class="tg-0pky">1-10.jsonl</td>
    <td class="tg-qdov">Economic History</td>
    <td class="tg-0pky"> </td>
    <td class="tg-qdov"> </td>
  </tr>
  <tr>
    <td class="tg-0pky">1-11.jsonl</td>
    <td class="tg-qdov">Finance</td>
    <td class="tg-0pky"> </td>
    <td class="tg-qdov"> </td>
  </tr>
  <tr>
    <td class="tg-lboi" rowspan="13"><b>Practice</b></td>
    <td class="tg-qdov">3-1.jsonl</td>
    <td class="tg-qdov">Junior Auditor</td>
    <td class="tg-lboi" rowspan="13"><b>Law</b></td>
    <td class="tg-qdov">4-1.jsonl</td>
    <td class="tg-qdov">Tax Law I</td>
  </tr>
  <tr>
    <td class="tg-0pky">3-2.jsonl</td>
    <td class="tg-qdov">Intermediate Auditor</td>
    <td class="tg-0pky">4-2.jsonl</td>
    <td class="tg-qdov">Tax Law II</td>
  </tr>
  <tr>
    <td class="tg-0pky">3-3.jsonl</td>
    <td class="tg-qdov">Junior Statistician</td>
    <td class="tg-0pky">4-3.jsonl</td>
    <td class="tg-qdov">Tax Inspection</td>
  </tr>
  <tr>
    <td class="tg-0pky">3-4.jsonl</td>
    <td class="tg-qdov">Intermediate Statistician</td>
    <td class="tg-0pky">4-4.jsonl</td>
    <td class="tg-qdov">Commercial Law</td>
  </tr>
  <tr>
    <td class="tg-0pky">3-5.jsonl</td>
    <td class="tg-qdov">Junior Economist</td>
    <td class="tg-0pky">4-5.jsonl</td>
    <td class="tg-qdov">Securities Law</td>
  </tr>
  <tr>
    <td class="tg-0pky">3-6.jsonl</td>
    <td class="tg-qdov">Intermediate Economist</td>
    <td class="tg-0pky">4-6.jsonl</td>
    <td class="tg-qdov">Insurance Law</td>
  </tr>
  <tr>
    <td class="tg-0pky">3-7.jsonl</td>
    <td class="tg-qdov">Junior Banking Professional</td>
    <td class="tg-0pky">4-7.jsonl</td>
    <td class="tg-qdov">Economic Law</td>
  </tr>
  <tr>
    <td class="tg-0pky">3-8.jsonl</td>
    <td class="tg-qdov">Intermediate Banking Professional</td>
    <td class="tg-0pky">4-8.jsonl</td>
    <td class="tg-qdov">Banking Law</td>
  </tr>
  <tr>
    <td class="tg-0pky">3-9.jsonl</td>
    <td class="tg-qdov">Junior Accountant</td>
    <td class="tg-0pky">4-9.jsonl</td>
    <td class="tg-qdov">Futures Law</td>
  </tr>
  <tr>
    <td class="tg-0pky">3-10.jsonl</td>
    <td class="tg-qdov">Intermediate Accountant</td>
    <td class="tg-0pky">4-10.jsonl</td>
    <td class="tg-qdov">Financial Law</td>
  </tr>
  <tr>
    <td class="tg-0pky">3-11.jsonl</td>
    <td class="tg-qdov">Tax Consultant</td>
    <td class="tg-0pky">4-11.jsonl</td>
    <td class="tg-qdov">Civil Law</td>
  </tr>
  <tr>
    <td class="tg-0pky">3-12.jsonl</td>
    <td class="tg-qdov">Asset Appraiser</td>
    <td class="tg-0pky"> </td>
    <td class="tg-qdov"> </td>
  </tr>
  <tr>
    <td class="tg-0pky">3-13.jsonl</td>
    <td class="tg-qdov">Securities Analyst</td>
    <td class="tg-0pky"> </td>
    <td class="tg-qdov"> </td>
  </tr>
</tbody>
</table>

## Evaluation

After preparing the datasets, you can evaluate the performance of the LLM models on CFinBench. Taking qwen1_5_7b and zero-shot setting as an example, you can run the following command:

```
python run.py --models hf_qwen1_5_7b --datasets cfinbench_gen_multi_judgment_zero-shot cevalcfinbench_gen_single_zero-shot --summarizer cfinbench_viz
```

The configuration details of the CFinBench can be referred to the files under `configs/datasets/CFinbench`, and the custom evaluator can be found in `opencompass/datasets/cfinbench.py`. 

## Benchmark Results

More detailed and comprehensive benchmark result can refer to our paper.

<div style="text-align: center;">

<img src="figs/result.png" alt="">
</div>

## Citation

If this project is beneficial to your research, please cite:

```
@misc{cFinBench,
      title={CFinBench: A Comprehensive Chinese Financial Benchmark for Large Language Models}, 
      author={Ying Nie, Binwei Yan, Tianyu Guo, Hao Liu, Haoyu Wang, Wei He and others},
      year={2024},
      eprint={},
}
```

## License

This project is released under the Apache 2.0 [license](LICENSE)