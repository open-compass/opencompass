# Multimodal Evaluation Overview

OpenCompass reuses multimodal datasets and official evaluators from [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) through a bridge. Dataset construction and scoring use the official VLMEvalKit implementation, while model inference, task scheduling, and result summarization use standard OpenCompass workflows. In addition to text configuration, multimodal evaluation involves image download locations, media blocks in messages, and image-input support in the model backend.

## Integration: Responsibilities of Three Components

The bridge consists of `VLMEvalKitDataset` on the data side, the standard inference pipeline, and `VLMEvalKitEvaluator` on the scoring side:

1. **Data loading:** `VLMEvalKitDataset.load` calls VLMEvalKit `build_dataset(dataset_name)` to construct the official dataset object. On first use, TSV data and images are downloaded under the `LMUData` cache according to official rules. Official `build_prompt()` then builds each input and the bridge converts it into structured OpenCompass messages: text blocks `{'type': 'text', ...}` and image blocks `{'type': 'image', 'image_url': <local path or URL>}`. Every sample also records `sample_id` (`<dataset name>:<index>`), the original row JSON, and reference answer.
2. **Inference:** the standard OpenCompass workflow is used. Dataset `infer_cfg` passes messages in the `prompt` column directly to the model using RawPromptTemplate `expand_column`, then `GenInferencer` generates normally. The backend must support image-content blocks. `OpenAI` / `OpenAISDK` currently do: local image paths are converted to base64 data URLs automatically, while `image_format` and `image_min_edge` control re-encoding and minimum resolution.
3. **Scoring:** `VLMEvalKitEvaluator` aligns predictions to the official table by `sample_id`, exports xlsx, and calls official `dataset.evaluate()`. It flattens returned aggregate metrics, converts them to percentages, and returns them to standard OpenCompass results and summarization.

The bridge has explicit boundaries and raises during construction otherwise:

- Only **IMAGE-modality** datasets are supported; video is unsupported.
- Only **single-turn** datasets are supported; official datasets marked as requiring multi-turn inference (`TYPE` is `MT`) are unsupported.
- A unique, nonempty `index` column is required.

## Installation

```bash
pip install "opencompass[vlm]"
```

The `vlm` extra installs libraries needed on the multimodal model side, including litellm and google-genai. VLMEvalKit itself must be installed separately according to its [official repository](https://github.com/open-compass/VLMEvalKit), so that `import vlmeval` works. Python **3.10+** is required; the bridge checks both requirements at startup.

## Running the Two Integrated Datasets

Configurations are currently provided for MMBench (DEV_EN) and MMMU-Pro (10c), together with complete examples:

- Dataset configurations: `opencompass/configs/datasets/MMBench/MMBench_DEV_EN_vlmevalkit_gen.py` and `opencompass/configs/datasets/MMMU_Pro/MMMU_Pro_10c_vlmevalkit_gen.py`.
- End-to-end examples: [examples/eval_mmbench_vlmevalkit.py](https://github.com/open-compass/opencompass/blob/main/examples/eval_mmbench_vlmevalkit.py) and [examples/eval_mmmu_pro_vlmevalkit.py](https://github.com/open-compass/opencompass/blob/main/examples/eval_mmmu_pro_vlmevalkit.py).

For MMBench, run the example directly:

```bash
export OPENAI_API_KEY=sk-xxx
opencompass examples/eval_mmbench_vlmevalkit.py
```

The complete workflow has four steps:

1. **Load:** `build_dataset` downloads/reads official TSV and images under `LMUData`, then generates messages with image blocks.
2. **Infer:** `expand_column` passes messages to the model. The example uses `OpenAISDK` against an OpenAI-compatible multimodal endpoint, automatically converting local images to base64.
3. **Score:** predictions are aligned to the official table, exported to xlsx, and given to official `evaluate()`. MMBench official scoring internally uses an LLM to extract choices, so the example sets `model`, `api_base`, `nproc`, `retry`, `timeout`, and related fields under `eval_cfg.evaluator.eval_kwargs`; they are forwarded unchanged to official scoring.
4. **Summarize:** metrics enter standard result files and summary tables.

For a small trial, limit samples through an environment variable:

```bash
MMBENCH_SAMPLE_LIMIT=20 opencompass examples/eval_mmbench_vlmevalkit.py
```

To use your own OpenAI-compatible multimodal service, change `path` and `openai_api_base` in the example model configuration and retain image arguments such as `image_format`. A language-model configuration cannot be substituted by changing only `type`; the model must actually accept image input.

## Data Cache and Environment Variable

| Environment variable | Purpose                                                        | Default                                            |
| -------------------- | -------------------------------------------------------------- | -------------------------------------------------- |
| `LMUData`            | VLMEvalKit data-cache root; TSV and images are downloaded here | `data/vlmevalkit` relative to the launch directory |

`LMUData` is VLMEvalKit's own data-directory convention. The dataset configuration reads it as `data_root`; during dataset construction and official scoring, the bridge temporarily points `LMUData` to this directory. Relative paths become absolute and are created automatically, ensuring that download, image reads, and scoring share the same data. On shared storage or in a container, explicitly set and mount it:

```bash
export LMUData=/shared/cache/vlmevalkit
```

## Reading Results

Assume `work_dir` is `outputs/mmbench_vlmevalkit` and the model abbreviation is `kimi-k2.6-chat-completions`:

- **Predictions:** `<work_dir>/predictions/<model abbr>/MMBench_DEV_EN.json`; each record contains original input messages including image references and the model reply.
- **Scoring artifacts:** `<work_dir>/results/<model abbr>/MMBench_DEV_EN.json` is the standard OpenCompass metric file. The sibling `MMBench_DEV_EN/` directory contains three official-protocol artifacts:
  - `MMBench_DEV_EN.xlsx`: complete prediction table aligned to official data, used directly by official scoring.
  - `vlmevalkit_evaluation.json`: snapshot of official scoring arguments including dataset name, data directory, and `eval_kwargs`, for reproduction.
  - `vlmevalkit_metrics.json`: flattened official aggregate metrics and the primary metric.
- **Summary:** CSV aggregate table under `<work_dir>/summary/`.

Metric names follow flattened official VLMEvalKit output, including group columns and Overall, and values are normalized to percentages. Dataset official logic determines the primary metric. Any empty sample prediction makes scoring fail because official scoring requires a complete prediction sequence.
