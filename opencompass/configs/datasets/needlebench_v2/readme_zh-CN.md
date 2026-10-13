# NeedleBench V2：改进版大海捞针测试评估基准

[English](readme.md) | 简体中文

## 概览

NeedleBench V2是一个改进版基准测试，旨在严格评估大型语言模型（LLMs）在长文本场景中的信息检索和推理能力。在原有NeedleBench的基础上，这个版本引入了重要的增强功能，为LLMs在海量文本中定位和推理关键信息的能力提供更准确、更公正的评估。

### 目录结构

```
configs/datasets/needlebench_v2/
├── atc
├── needlebench_v2_4k
├── needlebench_v2_8k
├── needlebench_v2_32k
├── needlebench_v2_128k
├── needlebench_v2_200k
├── needlebench_v2_256k
├── needlebench_v2_1000k
├── readme.md
└── readme_zh-CN.md
```

在每个长度配置目录下（如 `needlebench_v2_4k`），包含了专门针对该长度设置的测试任务配置文件。

## 任务描述与长度配置

NeedleBench V2提供了不同长度配置的任务（4k、8k、32k、128k、200k、256k、1000k），以适应不同规模的语言模型评估需求。每种长度配置针对以下任务提供了专门的测试脚本：

### 单针信息检索

单针信息检索任务评估LLMs从特定长度的无关信息文本中回忆单个重要信息的能力。这个任务评估模型在长文本中识别和回忆特定信息的精确性。

### 多针信息检索

多针信息检索任务挑战LLMs识别和提取广泛文本中的多个关键信息点的能力。它模拟了现实世界中的场景，其中需要从文档或报告中检索多个数据点、事实或数字，评估模型在浏览和从密集文本中提取相关信息的效率。

### 多针信息推理

在NeedleBench V2中，多针信息推理任务得到了显著改进。原来基于R4C/MultiHop数据集的"针"已被替换为类似于祖源追溯挑战中的虚构信息。这一改变解决了潜在的内生知识偏差问题，因为原始数据集可能已被包含在一些模型的训练数据中。这个任务继续评估LLMs使用检索到的信息进行复杂推理的能力，要求模型不仅能回忆多个信息点，还能进行逻辑推理。

### 祖源追溯挑战 (ATC)

祖源追溯挑战在NeedleBench V2中进行了优化。针的分布模式从密集形式（1、2、3、4、5针）变为基于2的幂次的稀疏形式（2¹、2²、2³等）。这个任务仍然是NeedleBench中最复杂的任务，要求模型回忆和分析长文本中的每个细节，以解决需要理解复杂关系的问题，如家谱查询或详细案例分析。

## 评分方法

NeedleBench V2引入了更平衡的评分系统。总体评分现在是通过三个主要任务（单针信息检索、多针信息检索和多针信息推理）的简单平均值计算得出，每个任务获得相等的权重。这一改变从先前的加权平均方法提供了一种更直接、更公平的方式，评估模型在不同检索和推理任务中的能力。

## 提示增强

NeedleBench V2中的所有提示都经过了改进，以提高清晰度和有效性，特别关注了ATC实验的提示。配置结构也进行了精简，使其更易于使用和理解。

## 引用

如果您在研究中使用NeedleBench V2，请引用：

```bibtex
@misc{li2025needlebenchllmsretrievalreasoning,
      title={NeedleBench: Can LLMs Do Retrieval and Reasoning in Information-Dense Context?}, 
      author={Mo Li and Songyang Zhang and Taolin Zhang and Haodong Duan and Yunxin Liu and Kai Chen},
      year={2025},
      eprint={2407.11963},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.11963}, 
}
``` 