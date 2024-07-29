# 安装

1. 准备 OpenCompass 运行环境：

`````{tabs}
````{tab} 面向开源模型的GPU环境

   ```bash
   conda create --name opencompass python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
   conda activate opencompass
   ```

   如果你希望自定义 PyTorch 版本或相关的 CUDA 版本，请参考 [官方文档](https://pytorch.org/get-started/locally/) 准备 PyTorch 环境。需要注意的是，OpenCompass 要求 `pytorch>=1.13`。

````

````{tab} 面向API模型测试的CPU环境

   ```bash
   conda create -n opencompass python=3.10 pytorch torchvision torchaudio cpuonly -c pytorch -y
   conda activate opencompass
   # 如果需要使用各个API模型，请 `pip install -r requirements/api.txt` 安装API模型的相关依赖
   ```

   如果你希望自定义 PyTorch 版本，请参考 [官方文档](https://pytorch.org/get-started/locally/) 准备 PyTorch 环境。需要注意的是，OpenCompass 要求 `pytorch>=1.13`。

````
`````

2. 安装 OpenCompass：

   ```bash
   git clone https://github.com/open-compass/opencompass.git
   cd opencompass
   pip install -e .
   ```

3. 如果需要使用推理后端，或者进行 API 模型测试，或者进行 代码、智能体、主观 等数据集的评测，请参考 [其他安装说明](./extra-installation.md)

## 数据集准备

OpenCompass 支持的数据集主要包括三个部分：

1. Huggingface 数据集： [Huggingface Dataset](https://huggingface.co/datasets) 提供了大量的数据集，这部分数据集运行时会**自动下载**。

2. ModelScope 数据集：[ModelScope OpenCompass Dataset](https://modelscope.cn/organization/opencompass) 支持从 ModelScope 自动下载数据集。

   要启用此功能，请设置环境变量：`export DATASET_SOURCE=ModelScope`，可用的数据集包括（来源于 OpenCompassData-core.zip）：

   ```plain
   humaneval, triviaqa, commonsenseqa, tydiqa, strategyqa, cmmlu, lambada, piqa, ceval, math, LCSTS, Xsum, winogrande, openbookqa, AGIEval, gsm8k, nq, race, siqa, mbpp, mmlu, hellaswag, ARC, BBH, xstory_cloze, summedits, GAOKAO-BENCH, OCNLI, cmnli
   ```

3. 自建以及第三方数据集：OpenCompass 还提供了一些第三方数据集及自建**中文**数据集。运行以下命令**手动下载解压**。

在 OpenCompass 项目根目录下运行下面命令，将数据集准备至 `${OpenCompass}/data` 目录下：

```bash
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip
```

如果需要使用 OpenCompass 提供的更加完整的数据集 (~500M)，可以使用下述命令进行下载和解压：

```bash
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-complete-20240207.zip
unzip OpenCompassData-complete-20240207.zip
cd ./data
find . -name "*.zip" -exec unzip "{}" \;
```

两个 `.zip` 中所含数据集列表如[此处](https://github.com/open-compass/opencompass/releases/tag/0.2.2.rc1)所示。

OpenCompass 已经支持了大多数常用于性能比较的数据集，具体支持的数据集列表请直接在 `configs/datasets` 下进行查找。

接下来，你可以阅读[快速上手](./quick_start.md)了解 OpenCompass 的基本用法。
