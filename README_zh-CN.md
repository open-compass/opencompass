<div align="center">
  <img src="https://user-images.githubusercontent.com/22607038/250798681-b52045d2-cedd-4070-84e2-410903ac404f.png" width="500px"/>

[![docs](https://readthedocs.org/projects/opencompass/badge/?version=dev-1.x)](https://opencompass.readthedocs.io/en/dev-1.x/?badge=dev-1.x)
[![license](https://img.shields.io/github/license/IntenLM/opencompass.svg)](https://github.com/InternLM/opencompass/blob/main/LICENSE)
[![PyPI](https://badge.fury.io/py/opencompass.svg)](https://pypi.org/project/opencompass/)

[📘Documentation](https://opencompass.readthedocs.io/en/latest/) |
[🛠️Installation](https://opencompass.readthedocs.io/en/latest/get_started/install.html) |
[🤔Reporting Issues](https://github.com/InternLM/opencompass/issues/new/choose)

[English](/README.md) | 简体中文

</div>

## 介绍

OpenCompass 是面向大模型评测的一站式平台，旨在提供一套公平、公开、可复现的大模型评测基准方案。其主要特点如下：

- **模型及数据集的全方位支持**：预支持 20+ HuggingFace 及 API 模型，并提供 50+ 个数据集约 30 万题的的模型评测方案，6 大维度的能力全面评测。

- **高效分布式评测**：一行命令实现任务分割和分布式评测，数小时即可完成千亿模型全量评测\*。

- **多样化评测范式**：支持零样本、小样本及思维链评测，结合标准型或对话型提示词模板，轻松激发各种模型最大性能。

- **易于扩展的模块化设计**：想增加新模型或数据集？想要自定义更高级的任务分割策略，甚至接入新的集群管理系统？OpenCompass 的一切均可轻松扩展！

- **完善的实验记录及上报机制**：使用配置文件完整记录每一次实验，关键信息有迹可循；结果实时上报飞书机器人，第一时间知晓实验情况。

## 模型能力排名

## 能力维度 & 模型支持

## 安装

下面展示了快速安装的步骤。有部分第三方功能可能需要额外步骤才能正常运行，详细步骤请参考[安装指南](https://opencompass.readthedocs.io/zh_cn/latest/get_started.html)。

```Python
conda create --name opencompass python=3.8 pytorch torchvision -c pytorch -y
conda activate opencompass
git clone https://github.com/InternLM/opencompass opencompass
cd opencompass
pip install -r requirements/runtime.txt
pip install -e .
# 下载数据集到 data/ 处
# TODO: ....
```

## 评测

请阅读[快速上手](https://opencompass.readthedocs.io/zh_cn/latest/get_started.html)了解如何运行一个评测任务。

## 致谢

该项目部分的代码引用并修改自 [OpenICL](https://github.com/Shark-NLP/OpenICL)。

## 引用

```bibtex
@misc{2023opencompass,
    title={OpenCompass: A Universal Evaluation Platform for Foundation Models},
    author={OpenCompass Contributors},
    howpublished = {\url{https://github.com/InternLM/OpenCompass}},
    year={2023}
}
```
