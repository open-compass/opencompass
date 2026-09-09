# 多模态评测总览

OpenCompass 通过桥接层复用 [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) 的多模态数据集与官方评测器：数据集构造和评分沿用 VLMEvalKit 的官方实现，模型推理、任务调度和结果汇总沿用 OpenCompass 的标准流程。多模态评测除了文本配置，还涉及图片资源的下载位置、消息中的媒体块以及模型后端的图片输入支持。

## 集成方式：三个组件的分工

桥接层由 `VLMEvalKitDataset`（数据集侧）、标准推理链路（推理侧）和 `VLMEvalKitEvaluator`（评分侧）组成：

1. **数据加载**：`VLMEvalKitDataset.load` 调用 VLMEvalKit 的 `build_dataset(dataset_name)` 构建官方数据集对象（首次运行会把 TSV 数据与图片按官方规则下载到 `LMUData` 缓存目录），再用官方 `build_prompt()` 生成每题输入，并转换成 OpenCompass 的结构化消息——文本块 `{'type': 'text', ...}` 与图片块 `{'type': 'image', 'image_url': <本地路径或 URL>}`。每条样本同时记录 `sample_id`（`<数据集名>:<index>`）、原始行 JSON 和标准答案。
2. **推理**：走 OpenCompass 标准流程。数据集的 `infer_cfg` 用 `RawPromptTemplate` 的 `expand_column` 直接把 `prompt` 列的消息透传给模型，`GenInferencer` 正常生成。模型后端必须支持图片内容块，目前 `OpenAI` / `OpenAISDK` 支持：本地图片路径会自动转成 base64 data URL，`image_format`、`image_min_edge` 参数可控制重编码方式与最小分辨率。
3. **评分**：`VLMEvalKitEvaluator` 把预测按 `sample_id` 对齐回官方数据表、导出 xlsx，然后调用官方 `dataset.evaluate()` 计算指标，把返回的汇总指标展平、换算成百分制后交回 OpenCompass 的结果与汇总体系。

桥接层有明确的适用边界（构建时会直接报错）：

- 只支持 **IMAGE 模态**数据集，视频类不支持；
- 只支持**单轮**数据集，官方标注需要多轮推理（TYPE 为 `MT`）的不支持；
- 要求数据集有唯一非空的 `index` 列。

## 安装

```bash
pip install "opencompass[vlm]"
```

`vlm` 附加依赖安装多模态模型侧所需库（litellm、google-genai 等）。VLMEvalKit 本体需另行安装（参照其[官方仓库](https://github.com/open-compass/VLMEvalKit)说明，使 `import vlmeval` 可用），且要求 **Python 3.10+**——桥接层启动时会做这两项检查。

## 运行已集成的两个数据集

当前提供 MMBench（DEV_EN）和 MMMU-Pro（10c）两个数据集配置，以及对应的完整示例：

- 数据集配置：`opencompass/configs/datasets/MMBench/MMBench_DEV_EN_vlmevalkit_gen.py`、`opencompass/configs/datasets/MMMU_Pro/MMMU_Pro_10c_vlmevalkit_gen.py`；
- 端到端示例：[examples/eval_mmbench_vlmevalkit.py](https://github.com/open-compass/opencompass/blob/main/examples/eval_mmbench_vlmevalkit.py)、[examples/eval_mmmu_pro_vlmevalkit.py](https://github.com/open-compass/opencompass/blob/main/examples/eval_mmmu_pro_vlmevalkit.py)。

以 MMBench 为例，直接运行示例配置：

```bash
export OPENAI_API_KEY=sk-xxx
opencompass examples/eval_mmbench_vlmevalkit.py
```

完整流程分四步：

1. **加载**：`build_dataset` 下载/读取官方 TSV 与图片到 `LMUData` 目录，为每题生成带图片块的消息；
2. **推理**：`expand_column` 把消息透传给模型（示例用 `OpenAISDK` 走 OpenAI 兼容多模态接口，本地图片自动转 base64）；
3. **评分**：预测对齐回官方表并导出 xlsx，交给官方 `evaluate()`。注意 MMBench 的官方评分内部会用 LLM 抽取选项，因此示例在 `eval_cfg.evaluator.eval_kwargs` 里配置了 `model`、`api_base`、`nproc`、`retry`、`timeout` 等——这组参数会原样传给官方评分函数；
4. **汇总**：指标进入标准结果文件和 summary 表。

试跑时可以用环境变量截取少量样本：

```bash
MMBENCH_SAMPLE_LIMIT=20 opencompass examples/eval_mmbench_vlmevalkit.py
```

被测模型换成自己的 OpenAI 兼容多模态服务时，参照示例修改模型配置中的 `path`、`openai_api_base`，并保留 `image_format` 等图片参数；不能用语言模型配置直接替换 `type`，模型必须真的接受图片输入。

## 数据缓存与环境变量

| 环境变量  | 作用                                              | 默认值                            |
| --------- | ------------------------------------------------- | --------------------------------- |
| `LMUData` | VLMEvalKit 的数据缓存根目录，TSV 与图片均下载于此 | `data/vlmevalkit`（相对启动目录） |

`LMUData` 是 VLMEvalKit 自身的数据目录约定：数据集配置在加载时读取它作为 `data_root`，桥接层在构建数据集和官方评分期间会把 `LMUData` 临时指向该目录（相对路径会转为绝对路径并自动创建），保证下载、读图和评分用的是同一份数据。共享存储或容器环境下，建议显式指定并挂载：

```bash
export LMUData=/shared/cache/vlmevalkit
```

## 结果怎么看

假设 `work_dir` 为 `outputs/mmbench_vlmevalkit`、模型 abbr 为 `kimi-k2.6-chat-completions`：

- **预测**：`<work_dir>/predictions/<模型 abbr>/MMBench_DEV_EN.json`，每条记录包含原始输入消息（含图片引用）和模型回复；
- **评分产物**：`<work_dir>/results/<模型 abbr>/MMBench_DEV_EN.json` 是 OpenCompass 的标准指标文件；同名目录 `MMBench_DEV_EN/` 下还有三个官方口径的产物：
  - `MMBench_DEV_EN.xlsx`：预测对齐回官方数据表后的完整预测表（官方评分的直接输入）；
  - `vlmevalkit_evaluation.json`：本次官方评分的参数快照（数据集名、数据目录、`eval_kwargs` 等），用于复现评分；
  - `vlmevalkit_metrics.json`：官方汇总指标的展平结果与主指标。
- **汇总**：`<work_dir>/summary/` 下的 CSV 汇总表。

指标名沿用 VLMEvalKit 官方汇总展平后的名称（包含各分组列与 Overall），数值已统一换算为百分制；哪个指标算主指标由数据集官方逻辑决定。任何样本预测为空都会让评分直接报错——官方评分要求完整的预测序列。
