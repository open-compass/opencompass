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
```
### 2. 推理和评测
```bash
cd ${path_to_opencompass}
mkdir -p data
ln -snf ${path_to_CHARM_repo}/data/CHARM ./data/CHARM

# 在CHARM上对模型hf_llama3_8b_instruct做推理和评测
python run.py --models hf_llama3_8b_instruct --datasets charm_gen
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
