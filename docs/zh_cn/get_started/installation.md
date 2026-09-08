# 安装与环境准备

OpenCompass 要求 Python 3.8 或更高版本。不同模型后端对 PyTorch、CUDA 和推理框架有各自的版本约束；准备 GPU 环境时，应先按模型与后端要求安装匹配的 PyTorch，再安装 OpenCompass。

## 创建独立环境

推荐使用 Python 3.12：

```bash
conda create -n opencompass python=3.12 -y
conda activate opencompass
```

OpenCompass 的常规安装和完整安装均支持 Python 3.12。但 APPS（`apps`、`apps_mini`）、TACO 和 LiveCodeBench Code Generation 等代码执行评测依赖 `pyext==0.7`。该库不兼容 Python 3.11 及之后的版本，因此如需运行这些评测，请创建 Python 3.10 环境。

LMDeploy 与 vLLM 可能要求不同版本的 PyTorch、CUDA 或其他依赖。需要使用多个推理后端时，建议为每个后端分别创建虚拟环境。

## 选择安装方式

只需评测常见语言模型和数据集时，可安装基础版本：

```bash
pip install -U opencompass
```

如需特定能力，可按用途安装相应的可选依赖：

```bash
pip install "opencompass[api]"       # OpenAI、Anthropic 等接口模型
pip install "opencompass[full]"      # 更多数据集与评测依赖
pip install "opencompass[vlm]"       # 多模态评测
pip install "opencompass[lmdeploy]"  # LMDeploy 后端
pip install "opencompass[vllm]"      # vLLM 后端
```

如需使用最新代码或参与开发，可从源码安装：

```bash
git clone https://github.com/open-compass/opencompass.git
cd opencompass
pip install -e .
```

源码安装后同样会注册 `opencompass` 命令；也可以在仓库根目录使用等价的 `python run.py` 入口。

## 验证安装

以下命令适用于 PyPI 安装和源码安装：`which python` 用于确认当前 Python 环境，随后检查 OpenCompass 的版本和实际导入路径，最后验证命令行入口及其依赖是否可用。

```bash
which python
python -c "import opencompass; print(opencompass.__version__); print(opencompass.__file__)"
opencompass --help
```

## 数据缓存

数据集通常会在首次使用时下载。在共享机器上，可以通过以下环境变量指定缓存目录：

```bash
export HF_DATASETS_CACHE=/path/to/huggingface-cache/datasets
export COMPASS_DATA_CACHE=/path/to/opencompass-data-cache
```

其中，`HF_DATASETS_CACHE` 用于管理 Hugging Face 数据集缓存，`COMPASS_DATA_CACHE` 用于指定 OpenCompass 的数据缓存根目录。不同数据集的下载和读取方式可能不同，详细规则及离线环境准备方法请参阅[数据来源、缓存与离线运行](../user_guides/data_and_cache.md)。

## 检查模型及推理后端

成功安装 LMDeploy 或 vLLM 并不代表目标模型一定与后端兼容。建议先使用相应后端加载一个小模型，再通过 OpenCompass 运行演示数据集。使用 API 模型时，还应检查服务地址、模型名称、密钥环境变量、限流设置和超时配置。

## 开始评测！

安装完成后，请继续阅读[五分钟快速开始](quick_start.md)。遇到依赖、显存或下载问题时，请参阅[常见问题与故障排查](../faq/index.md)。
