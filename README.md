# Custom OpenCompass Extension

This repository is a fork of [OpenCompass](https://github.com/open-compass/opencompass), extended with **custom model configurations** and **dataset settings** for evaluation experiments.

We add support for additional models under:

```
opencompass/configs/models/a_test_models_instruct/
```

and custom datasets under:

```
opencompass/configs/datasets/
```

---

## 🛠️ Installation

Please follow the official [OpenCompass installation guide](https://doc.opencompass.org.cn/get_started/installation.html) to set up the environment.

> You may also refer to the customized version of the original README provided here as `README-opencompass.md`.

---

## 🚀 Usage

### 1. Dataset Preparation

Before running any evaluation, please download the necessary datasets according to the [OpenCompass documentation](https://doc.opencompass.org.cn/get_started/installation.html#dataset-preparation).

### 2. Model Path Configuration

After downloading the required models, **update the local model path** in the configuration files under:

```
opencompass/configs/models/a_test_models_instruct/
```

For example:

```python
models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr="gemma-2-2b-instruct-vllm",
        path="chengxiang/datas/models/gemma2-2B-it",  # 🔧 Replace this with the actual local model path
        model_kwargs=dict(
            tensor_parallel_size=4,
            max_model_len=8192,
            gpu_memory_utilization=0.9
        ),
        max_out_len=4096,
        max_seq_len=8192,
        batch_size=10000,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=4),
        meta_template=meta_template
    )
]
```

### 3. Running Evaluations

You may run evaluations using the same commands provided in the official documentation. Additionally, this repo includes **custom evaluation scripts** in:

```
opencompass/eval_scripts/
```

Feel free to explore and adapt them for your experiments.

---

## 📚 Notes

* Custom dataset configurations are added under `opencompass/configs/datasets/`, with a focus on the **GSM8K** and **MATH** datasets.

* These configurations explore a wide variety of **prompting settings**, located respectively in:

  * `a_gsm8k/` for GSM8K
  * `a_math/` for MATH

These files are designed for flexible experimentation and can serve as useful references for creating your own dataset settings.

---

## 📎 References

* [OpenCompass GitHub Repository](https://github.com/open-compass/opencompass)
* Local documentation: `README-opencompass.md`


