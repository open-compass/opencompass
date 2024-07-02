# CHARM✨ Benchmarking Chinese Commonsense Reasoning of LLMs: From Chinese-Specifics to Reasoning-Memorization Correlations [ACL2024]
[![arXiv](https://img.shields.io/badge/arXiv-2403.14112-b31b1b.svg)](https://arxiv.org/abs/2403.14112)
[![license](https://img.shields.io/github/license/InternLM/opencompass.svg)](./LICENSE)
<div align="center">

📃[Paper](https://arxiv.org/abs/2403.14112)
🏰[Project Page](https://opendatalab.github.io/CHARM/)
🏆[Leaderboard](https://opendatalab.github.io/CHARM/leaderboard.html)
✨[Findings](https://opendatalab.github.io/CHARM/findings.html)

</div>

<div align="center">
    📖 <a href="./README_ZH.md">   中文</a> | <a href="./README.md">English</a>
</div>

## Dataset Description

**CHARM** is the first benchmark for comprehensively and in-depth evaluating the commonsense reasoning ability of large language models (LLMs) in Chinese, which covers both globally known and Chinese-specific commonsense. In addition, the CHARM can evaluate the LLMs' memorization-independent reasoning abilities and analyze the typical errors.

## Comparison of commonsense reasoning benchmarks
<html lang="en">
        <table align="center">
            <thead class="fixed-header">
                <tr>
                    <th>Benchmarks</th>
                    <th>CN-Lang</th>
                    <th>CSR</th>
                    <th>CN-specifics</th>
                    <th>Dual-Domain</th>
                    <th>Rea-Mem</th>
                </tr>
            </thead>
            <tr>
                <td>Most benchmarks in <a href="https://arxiv.org/abs/2302.04752"> davis2023benchmarks</a></td>
                <td><strong><span style="color: red;">&#x2718;</span></strong></td>
                <td><strong><span style="color: green;">&#x2714;</span></strong></td>
                <td><strong><span style="color: red;">&#x2718;</span></strong></td>
                <td><strong><span style="color: red;">&#x2718;</span></strong></td>
                <td><strong><span style="color: red;">&#x2718;</span></strong></td>
            </tr>
            <tr>
                <td><a href="https://arxiv.org/abs/1809.05053"> XNLI</a>, <a
                        href="https://arxiv.org/abs/2005.00333">XCOPA</a>,<a
                        href="https://arxiv.org/abs/2112.10668">XStoryCloze</a></td>
                <td><strong><span style="color: green;">&#x2714;</span></strong></td>
                <td><strong><span style="color: green;">&#x2714;</span></strong></td>
                <td><strong><span style="color: red;">&#x2718;</span></strong></td>
                <td><strong><span style="color: red;">&#x2718;</span></strong></td>
                <td><strong><span style="color: red;">&#x2718;</span></strong></td>
            </tr>
            <tr>
                <td><a href="https://arxiv.org/abs/2007.08124">LogiQA</a>, <a
                        href="https://arxiv.org/abs/2004.05986">CLUE</a>, <a
                        href="https://arxiv.org/abs/2306.09212">CMMLU</a></td>
                <td><strong><span style="color: green;">&#x2714;</span></strong></td>
                <td><strong><span style="color: red;">&#x2718;</span></strong></td>
                <td><strong><span style="color: green;">&#x2714;</span></strong></td>
                <td><strong><span style="color: red;">&#x2718;</span></strong></td>
                <td><strong><span style="color: red;">&#x2718;</span></strong></td>
            </tr>
            <tr>
                <td><a href="https://arxiv.org/abs/2312.12853">CORECODE</a> </td>
                <td><strong><span style="color: green;">&#x2714;</span></strong></td>
                <td><strong><span style="color: green;">&#x2714;</span></strong></td>
                <td><strong><span style="color: red;">&#x2718;</span></strong></td>
                <td><strong><span style="color: red;">&#x2718;</span></strong></td>
                <td><strong><span style="color: red;">&#x2718;</span></strong></td>
            </tr>
            <tr>
                <td><strong><a href="https://arxiv.org/abs/2403.14112">CHARM (ours)</a> </strong></td>
                <td><strong><span style="color: green;">&#x2714;</span></strong></td>
                <td><strong><span style="color: green;">&#x2714;</span></strong></td>
                <td><strong><span style="color: green;">&#x2714;</span></strong></td>
                <td><strong><span style="color: green;">&#x2714;</span></strong></td>
                <td><strong><span style="color: green;">&#x2714;</span></strong></td>
            </tr>
        </table>

"CN-Lang" indicates the benchmark is presented in Chinese language. "CSR" means the benchmark is designed to focus on <strong>C</strong>ommon<strong>S</strong>ense <strong>R</strong>easoning. "CN-specific" indicates the benchmark includes elements that are unique to Chinese culture, language, regional characteristics, history, etc. "Dual-Domain" indicates the benchmark encompasses both Chinese-specific and global domain tasks, with questions presented in the similar style and format. "Rea-Mem" indicates the benchmark includes closely-interconnected <strong>rea</strong>soning and <strong>mem</strong>orization tasks.


## 🛠️ How to Use
Below are the steps for quickly downloading CHARM and using OpenCompass for evaluation.

### 1. Download CHARM
```bash
git clone https://github.com/opendatalab/CHARM ${path_to_CHARM_repo}

cd ${path_to_opencompass}
mkdir data
ln -snf ${path_to_CHARM_repo}/data/CHARM ./data/CHARM
```
### 2. Run Inference and Evaluation
```bash
cd ${path_to_opencompass}

# modify config file `configs/eval_charm_rea.py`: uncomment or add models you want to evaluate
python run.py configs/eval_charm_rea.py -r --dump-eval-details

# modify config file `configs/eval_charm_mem.py`: uncomment or add models you want to evaluate
python run.py configs/eval_charm_mem.py -r --dump-eval-details
```
The inference and evaluation results would be in `${path_to_opencompass}/outputs`, like this:
```bash
outputs
├── CHARM_mem
│   └── chat
│       └── 20240605_151442
│           ├── predictions
│           │   ├── internlm2-chat-1.8b-turbomind
│           │   ├── llama-3-8b-instruct-lmdeploy
│           │   └── qwen1.5-1.8b-chat-hf
│           ├── results
│           │   ├── internlm2-chat-1.8b-turbomind_judged-by--GPT-3.5-turbo-0125
│           │   ├── llama-3-8b-instruct-lmdeploy_judged-by--GPT-3.5-turbo-0125
│           │   └── qwen1.5-1.8b-chat-hf_judged-by--GPT-3.5-turbo-0125
│           └── summary
│               └── 20240605_205020 # MEMORY_SUMMARY_DIR
│                   ├── judged-by--GPT-3.5-turbo-0125-charm-memory-Chinese_Anachronisms_Judgment
│                   ├── judged-by--GPT-3.5-turbo-0125-charm-memory-Chinese_Movie_and_Music_Recommendation
│                   ├── judged-by--GPT-3.5-turbo-0125-charm-memory-Chinese_Sport_Understanding
│                   ├── judged-by--GPT-3.5-turbo-0125-charm-memory-Chinese_Time_Understanding
│                   └── judged-by--GPT-3.5-turbo-0125.csv # MEMORY_SUMMARY_CSV
└── CHARM_rea
    └── chat
        └── 20240605_152359
            ├── predictions
            │   ├── internlm2-chat-1.8b-turbomind
            │   ├── llama-3-8b-instruct-lmdeploy
            │   └── qwen1.5-1.8b-chat-hf
            ├── results # REASON_RESULTS_DIR
            │   ├── internlm2-chat-1.8b-turbomind
            │   ├── llama-3-8b-instruct-lmdeploy
            │   └── qwen1.5-1.8b-chat-hf
            └── summary
                ├── summary_20240605_205328.csv # REASON_SUMMARY_CSV
                └── summary_20240605_205328.txt
```
### 3. Generate Analysis Results
```bash
cd ${path_to_CHARM_repo}

# generate Table5, Table6, Table9 and Table10 in https://arxiv.org/abs/2403.14112
PYTHONPATH=. python tools/summarize_reasoning.py ${REASON_SUMMARY_CSV}

# generate Figure3 and Figure9 in https://arxiv.org/abs/2403.14112
PYTHONPATH=. python tools/summarize_mem_rea.py ${REASON_SUMMARY_CSV} ${MEMORY_SUMMARY_CSV}

# generate Table7, Table12, Table13 and Figure11 in https://arxiv.org/abs/2403.14112
PYTHONPATH=. python tools/analyze_mem_indep_rea.py data/CHARM ${REASON_RESULTS_DIR} ${MEMORY_SUMMARY_DIR} ${MEMORY_SUMMARY_CSV}
```

## 🖊️ Citation
```bibtex
@misc{sun2024benchmarking,
      title={Benchmarking Chinese Commonsense Reasoning of LLMs: From Chinese-Specifics to Reasoning-Memorization Correlations},
      author={Jiaxing Sun and Weiquan Huang and Jiang Wu and Chenya Gu and Wei Li and Songyang Zhang and Hang Yan and Conghui He},
      year={2024},
      eprint={2403.14112},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
