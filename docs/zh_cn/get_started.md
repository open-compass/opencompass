# 概述

# 安装

1. 参考 [PyTorch](https://pytorch.org/) 准备 Torch。

注意，OpenCompass 需要 `pytorch>=1.13`。

```bash
conda create --name opencompass python=3.8 -y
conda activate opencompass
conda install pytorch torchvision -c pytorch
```

2. 安装 OpenCompass：

```bash
git clone https://github.com/opencompass/opencompass
cd opencompass
pip install -r requirments/runtime.txt
pip install -e .
```

3. 安装 humaneval（可选）

如果你希望在 humaneval 数据集上进行评估，请执行此步骤。

```
git clone https://github.com/openai/human-eval.git
cd human-eval
pip install -r requirments.txt
pip install -e .
```

请记住在源代码中删除第48-57行的注释，并取消对[第58行](https://github.com/openai/human-eval/blob/312c5e5532f0e0470bf47f77a6243e02a61da530/human_eval/execution.py#L58)的注释。

# 快速上手

