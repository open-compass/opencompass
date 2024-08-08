# Needlebench：大海捞针测试评估基准

[English](readme.md) | 简体中文

## 概览

Needlebench是一个全面的基准测试，旨在严格评估大型语言模型（LLMs）的信息检索和推理能力。借鉴了NeedleInAHaystack实验的灵感，Needlebench扩展了范围，包括多种任务，每个任务都旨在测试LLMs处理长文本中关键信息的不同方面的能力。

### 目录结构

```
configs/datasets/needlebench/
├── atc
├── needlebench_4k
├── needlebench_8k
├── needlebench_32k
├── needlebench_128k
├── needlebench_200k
├── needlebench_1000k
├── needlebench.py
├── readme.md
└── readme_zh-CN.md
```

在每个长度配置目录下（如 `needlebench_4k`），包含了专门针对该长度设置的测试任务脚本：

```
needlebench_4k/
├── needlebench_multi_reasoning.py
├── needlebench_multi_retrieval.py
├── needlebench.py
└── needlebench_single.py
```

## 任务描述与长度配置

Needlebench提供了不同长度配置的任务，如4k、8k等，以适应不同规模的语言模型评估需求。每种长度配置针对以下任务提供了专门的测试脚本：

### 单针信息检索 (`needlebench_single.py`)

单针信息检索任务评估LLMs从特定长度的无关信息文本中回忆单个重要信息的能力。这个任务反映了原始的NeedleInAHaystack测试的目标，评估模型长文本中识别和回忆特定信息的精确性。

### 多针信息检索 (`needlebench_multi_retrieval.py`)

多针信息检索任务挑战LLMs识别和提取广泛文本中的多个关键信息点的能力。它模拟了现实世界中的场景，其中需要从文档或报告中检索多个数据点、事实或数字，评估模型在浏览和从密集文本中提取相关信息的效率。

### 多针信息推理 (`needlebench_multi_reasoning.py`)

在检索任务的基础上，多针信息推理任务强调LLMs使用检索到的信息进行复杂推理的能力。模型不仅需要回忆多个信息点，还需要进行逻辑推理，综合回答反映对不同信息点之间复杂关系理解的答案。

### 祖源追溯挑战 (ATC)

祖源追溯挑战是Needlebench中最复杂的任务，要求模型回忆和分析长文本中的每个细节，以解决需要理解复杂关系的问题，如家谱查询或详细案例分析。这个任务突出了模型处理和推理详细信息的需要，反映了现实世界中对复杂实际任务的要求。
