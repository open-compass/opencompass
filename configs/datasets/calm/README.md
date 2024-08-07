# CaLM Lite
**CaLM Lite** is a lightweight version of CaLM.

**Ca**usal evaluation of **L**anguage **M**odels (CaLM), to the best of our knowledge, is the first comprehensive benchmark for evaluating the causal reasoning capabilities of language models. The CaLM framework establishes a foundational taxonomy consisting of four modules: causal target (i.e., what to evaluate), adaptation (i.e., how to obtain the results), metric (i.e., how to measure the results), and error (i.e., how to analyze the bad results).

<div align="center">

[🌐 Website](https://opencausalab.github.io/CaLM) |
[📃 Report](https://arxiv.org/abs/2405.00622) |[ 🎆 Github](https://github.com/OpenCausaLab/CaLM) | 📧 Welcome to join us by email at causalai@pjlab.org.cn
</div>

## Quick Start
### Data Preparation
Download dataset to data/ folder.
```
wget https://github.com/OpenCausaLab/CaLM/releases/download/v1.0.0.lite/calm.zip
unzip calm.zip
```
### Run Model and Infer
To obtain a concise output with only the average information for all tasks, use:

```
python run.py --models YOUR_MODEL --datasets calm --summarizer calm
```

If you want detailed information for each task, use:

```
python run.py --models YOUR_MODEL --datasets calm
```

The `--summarizer calm` flag in the first command is used to generate a summarized output, while omitting it in the second command will provide task-specific details.
## Available Causal Tasks
We provide 92 tasks for causal evaluation, stored in the `data/calm` folder. For more information about our causal tasks, refer to [tasks](https://github.com/OpenCausaLab/CaLM/blob/main/documents/tasks.md).
The directory structure is:

```
├── calm
| ├── association
| ├── causal_discovery # Rung of the causal ladder
| │ ├── abstract_reasoning # Causal scenario
| │ │ ├── AR-B_CaLM-AR_CN.json # Causal task
| │ | └── AR-B_CaLM-AR_EN.json # Causal task
| │ └── ...
| └── ...
└── ...
```

## Dataset
- **Dataset size**: CaLM Lite leverages a light dataset of **9200**, while CaLM uses a significantly larger dataset of 126,334. The table below details the English dataset composition, with the Chinese version structured identically.
- **Dataset configuration**: We prioritize balance in our dataset for **binary classification** and **choice selection** questions. By ensuring an equal number of each GT label, we minimize the risk of introducing bias into the model's testing. For **probability calculation**, CaLM-Lite takes extra attention to balance the number of problems across different causal reasoning processes. (For more details on how causal reasoning process is defined, please refer to Section 9.1.6 of the [paper](https://arxiv.org/abs/2405.00622).)
- **Efficient evaluation**: For enhanced evaluation efficiency, OpenCompass offers customizable methods. Refer to the [documentation](https://opencompass.org.cn/doc) for guidance on tailoring these methods to your needs.

| Causal ladder | Causal scenario | Subset | Question type | Mode | CaLM Lite | CaLM |
|---------------|-----------------|--------|---------------|------|-----------|------|
| Causal discovery | PCD | E-CARE | Binary classification | Natural | 100 | 2000 |
| Causal discovery | PCD | E-CARE | Choice selection | Natural | 100 | 1000 |
| Causal discovery | PCD | COPA | Binary classification | Natural | 100 | 2000 |
| Causal discovery | PCD | COPA | Choice selection | Natural | 100 | 1000 |
| Causal discovery | ECI | CTB | Binary classification | Natural | 100 | 596 |
| Causal discovery | ECI | ESC | Binary classification | Natural | 100 | 1000 |
| Causal discovery | ECI | MAVEN-ERE | Binary classification | Natural | 100 | 1000 |
| Causal discovery | AR | CaLM-AR | Binary classification | Symbolic | 100 | 1600 |
| Causal discovery | CA | FP | Binary classification | Symbolic | 100 | 1600 |
| Causal discovery | CA | FA | Binary classification | Symbolic | 100 | 1600 |
| Association | CORR | correlation | Binary classification | Natural | 100 | 1476 |
| Association | EAE | exp-away | Binary classification | Natural | 100 | 168 |
| Intervention | CB | collider-bias | Binary classification | Natural | 100 | 163 |
| Intervention | ATE | ATE-natural | Binary classification | Natural | 100 | 1600 |
| Intervention | ATE | ATE-basic | Probability calculation | Mathematical | 100 | 1600 |
| Intervention | ATE | ATE-hard | Probability calculation | Mathematical | 100 | 1600 |
| Intervention | CDE | CDE-natural | Binary classification | Natural | 100 | 1600 |
| Intervention | CDE | CDE-basic | Probability calculation | Mathematical | 100 | 1600 |
| Intervention | CDE | CDE-hard | Probability calculation | Mathematical | 100 | 1600 |
| Intervention | BAS | backadj | Binary classification | Natural | 100 | 227 |
| Intervention | BAS | max-BAS | Choice selection | Symbolic | 100 | 1600 |
| Intervention | BAS | min-BAS | Choice selection | Symbolic | 100 | 1600 |
| Intervention | BAS | mix-BAS | Choice selection | Symbolic | 100 | 1600 |
| Intervention | FAS | FAS | Choice selection | Symbolic | 100 | 1600 |
| Intervention | IV | CaLM-IV | Choice selection | Symbolic | 100 | 1600 |
| Intervention | CEI | 0.2-UC | Binary classification | Symbolic | 100 | 1600 |
| Intervention | CEI | 0.4-UC | Binary classification | Symbolic | 100 | 1600 |
| Intervention | CEI | 0.6-UC | Binary classification | Symbolic | 100 | 1600 |
| Intervention | CEI | 0.8-UC | Binary classification | Symbolic | 100 | 1600 |
| Counterfactuals | ETT | ETT-natural | Binary classification | Natural | 100 | 1600 |
| Counterfactuals | ETT | ETT-basic | Probability calculation | Mathematical | 100 | 1600 |
| Counterfactuals | ETT | ETT-hard | Probability calculation | Mathematical | 100 | 1600 |
| Counterfactuals | NDE | NDE-natural | Binary classification | Natural | 100 | 1600 |
| Counterfactuals | NDE | NDE-basic | Probability calculation | Mathematical | 100 | 1600 |
| Counterfactuals | NDE | NDE-hard | Probability calculation | Mathematical | 100 | 1600 |
| Counterfactuals | NIE | NIE-natural | Binary classification | Natural | 100 | 1600 |
| Counterfactuals | NIE | NIE-basic | Probability calculation | Mathematical | 100 | 1600 |
| Counterfactuals | NIE | NIE-hard | Probability calculation | Mathematical | 100 | 1600 |
| Counterfactuals | PN | PN-basic | Probability calculation | Mathematical | 100 | 1600 |
| Counterfactuals | PN | PN-hard | Probability calculation | Mathematical | 100 | 1600 |
| Counterfactuals | PS | PS-basic | Probability calculation | Mathematical | 100 | 1600 |
| Counterfactuals | PS | PS-hard | Probability calculation | Mathematical | 100 | 1600 |
| Counterfactuals | AC | causal judgement | Binary classification | Natural | 100 | 187 |
| Counterfactuals | CR | CRASS | Choice selection | Natural | 100 | 274 |
| Counterfactuals | CR | det-counterfactual | Binary classification | Natural | 100 | 1476 |
| Counterfactuals | CEG | E-CARE | Open-ended generation | Natural | 100 | 1000 |
| **Total** | | | | | 4600 | 63167 |

## Available Prompt Styles (Adaptation)
Basic Prompt is our default setting for efficient evaluation of CaLM Lite, but we provide flexibility for exploring additional prompts through CaLM. If you'd like to explore and compare a wider range of prompts, we encourage you to use CaLM. We provide a comprehensive and easy-to-follow guide to assist you in our [repository](https://github.com/OpenCausaLab/CaLM).

## Citation
```
@misc{chen2024causal,
      title={Causal Evaluation of Language Models},
      author={Sirui Chen and Bo Peng and Meiqi Chen and Ruiqi Wang and Mengying Xu and Xingyu Zeng and Rui Zhao and Shengjie Zhao and Yu Qiao and Chaochao Lu},
      year={2024},
      eprint={2405.00622},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
