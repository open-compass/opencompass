# Overview

# Installation

1. Prepare Torch refer to [PyTorch](https://pytorch.org/). 

Notice that OpenCompass requires `pytorch>=1.13`.

```bash
conda create --name opencompass python=3.8 -y
conda activate opencompass
conda install pytorch torchvision -c pytorch
```

2. Install OpenCompass:

```bash
git clone https://github.com/opencompass/opencompass
cd opencompass
pip install -r requirments/runtime.txt
pip install -e .
```

3. Install humaneval (option) 

do this if you want to eval on humaneval dataset.

```
git clone https://github.com/openai/human-eval.git
cd human-eval
pip install -r requirments.txt
pip install -e .
```

Remember to remove the comments of Line48-57 and uncomment [line58](https://github.com/openai/human-eval/blob/312c5e5532f0e0470bf47f77a6243e02a61da530/human_eval/execution.py#L58) in the source code.

# Quick tour



