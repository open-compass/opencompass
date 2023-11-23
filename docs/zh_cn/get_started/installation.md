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

3. 安装 humaneval（可选）：

   如果你需要**在 humaneval 数据集上评估模型代码能力**，请执行此步骤，否则忽略这一步。

   <details>
   <summary><b>点击查看详细</b></summary>

   ```bash
   git clone https://github.com/openai/human-eval.git
   cd human-eval
   pip install -r requirements.txt
   pip install -e .
   cd ..
   ```

   请仔细阅读 `human_eval/execution.py` **第48-57行**的注释，了解执行模型生成的代码可能存在的风险，如果接受这些风险，请取消**第58行**的注释，启用代码执行评测。

   </details>

4. 安装 Llama（可选）：

   如果你需要**使用官方实现评测 Llama / Llama-2 / Llama-2-chat 模型**，请执行此步骤，否则忽略这一步。

   <details>
   <summary><b>点击查看详细</b></summary>

   ```bash
   git clone https://github.com/facebookresearch/llama.git
   cd llama
   pip install -r requirements.txt
   pip install -e .
   cd ..
   ```

   你可以在 `configs/models` 下找到所有 Llama / Llama-2 / Llama-2-chat 模型的配置文件示例。([示例](https://github.com/open-compass/opencompass/blob/eb4822a94d624a4e16db03adeb7a59bbd10c2012/configs/models/llama2_7b_chat.py))

   </details>

# 数据集准备

OpenCompass 支持的数据集主要包括两个部分：

1. Huggingface 数据集： [Huggingface Dataset](https://huggingface.co/datasets) 提供了大量的数据集，这部分数据集运行时会**自动下载**。

2. 自建以及第三方数据集：OpenCompass 还提供了一些第三方数据集及自建**中文**数据集。运行以下命令**手动下载解压**。

在 OpenCompass 项目根目录下运行下面命令，将数据集准备至 `${OpenCompass}/data` 目录下：

```bash
wget https://github.com/open-compass/opencompass/releases/download/0.1.8.rc1/OpenCompassData-core-20231110.zip
unzip OpenCompassData-core-20231110.zip
```

如果需要使用 OpenCompass 提供的更加完整的数据集 (~500M)，可以使用下述命令进行下载：

```bash
wget https://github.com/open-compass/opencompass/releases/download/0.1.8.rc1/OpenCompassData-complete-20231110.zip
unzip OpenCompassData-complete-20231110.zip
cd ./data
unzip *.zip
```

两个 `.zip` 中所含数据集列表如[此处](https://github.com/open-compass/opencompass/releases/tag/0.1.8.rc1)所示。

OpenCompass 已经支持了大多数常用于性能比较的数据集，具体支持的数据集列表请直接在 `configs/datasets` 下进行查找。

接下来，你可以阅读[快速上手](./quick_start.md)了解 OpenCompass 的基本用法。
