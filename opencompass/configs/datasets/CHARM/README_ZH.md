# CHARM✨ Benchmarking Chinese Commonsense Reasoning of LLMs: From Chinese-Specifics to Reasoning-Memorization Correlations [ACL2024]
[![arXiv](https://img.shields.io/badge/arXiv-2403.14112-b31b1b.svg)](https://arxiv.org/abs/2403.14112)
[![license](https://img.shields.io/github/license/InternLM/opencompass.svg)](../../../../LICENSE)
<div align="center">

📃[Paper](https://arxiv.org/abs/2403.14112)
🏰[Project Page](https://opendatalab.github.io/CHARM/)
🏆[Leaderboard](https://opendatalab.github.io/CHARM/leaderboard.html)
✨[Findings](https://opendatalab.github.io/CHARM/findings.html)
</div>

<div align="center">
    📖 <a href="./README_ZH.md">   中文</a> | <a href="./README.md">English</a>
</div>

## 数据集介绍

**CHARM** 是首个全面深入评估大型语言模型（LLMs）在中文常识推理能力的基准测试，它覆盖了国际普遍认知的常识以及独特的中国文化常识。此外，CHARM 还可以评估 LLMs 独立于记忆的推理能力，并分析其典型错误。


## 与其他常识推理评测基准的比较
<html lang="en">
        <table align="center">
            <thead class="fixed-header">
                <tr>
                    <th>基准</th>
                    <th>汉语</th>
                    <th>常识推理</th>
                    <th>中国特有知识</th>
                    <th>中国和世界知识域</th>
                    <th>推理和记忆的关系</th>
                </tr>
            </thead>
            <tr>
                <td><a href="https://arxiv.org/abs/2302.04752"> davis2023benchmarks</a> 中提到的基准</td>
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
                <td><a href="https://arxiv.org/abs/2007.08124">LogiQA</a>,<a
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


## 🛠️ 如何使用
以下是快速下载 CHARM 并在 OpenCompass 上进行评估的步骤。

### 1. 下载 CHARM
```bash
git clone https://github.com/opendatalab/CHARM ${path_to_CHARM_repo}

cd ${path_to_opencompass}
mkdir data
ln -snf ${path_to_CHARM_repo}/data/CHARM ./data/CHARM
```
### 2. 推理和评测
```bash
cd ${path_to_opencompass}

# 修改配置文件`examples/eval_charm_rea.py`: 将现有的模型取消注释，或者添加你想评测的模型
python run.py examples/eval_charm_rea.py -r --dump-eval-details

# 修改配置文件`examples/eval_charm_mem.py`: 将现有的模型取消注释，或者添加你想评测的模型
python run.py examples/eval_charm_mem.py -r --dump-eval-details
```
推理和评测的结果位于路径`${path_to_opencompass}/outputs`, 如下所示:
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
### 3. 生成分析结果
```bash
cd ${path_to_CHARM_repo}

# 生成论文中的Table5, Table6, Table9 and Table10，详见https://arxiv.org/abs/2403.14112
PYTHONPATH=. python tools/summarize_reasoning.py ${REASON_SUMMARY_CSV}

# 生成论文中的Figure3 and Figure9，详见https://arxiv.org/abs/2403.14112
PYTHONPATH=. python tools/summarize_mem_rea.py ${REASON_SUMMARY_CSV} ${MEMORY_SUMMARY_CSV}

# 生成论文中的Table7, Table12, Table13 and Figure11，详见https://arxiv.org/abs/2403.14112
PYTHONPATH=. python tools/analyze_mem_indep_rea.py data/CHARM ${REASON_RESULTS_DIR} ${MEMORY_SUMMARY_DIR} ${MEMORY_SUMMARY_CSV}
```

## 🖊️ 引用
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
