# Needle In A Haystack Evaluation

## Introduction to the Needle In A Haystack Test

The Needle In A Haystack test (inspired by [NeedleInAHaystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack/blob/main/LLMNeedleHaystackTester.py)) is an evaluation method where key information is randomly inserted into long texts to form the prompt for large language models (LLMs). This test aims to assess whether LLMs can extract critical information from long texts, thereby evaluating their fundamental ability to comprehend and process long-context documents.

## Task Overview

Within the `OpenCompass` framework, under `NeedleBench`, we designed a series of progressively challenging evaluation tasks to comprehensively assess LLMs' long-text information extraction and reasoning capabilities. For a complete description, please refer to our [technical report](https://arxiv.org/abs/2407.11963).

- **Single-Needle Retrieval Task (S-RT)**: Evaluates the LLM's ability to retrieve a single piece of key information from a long text, testing precise recall of specific details within extensive narratives. This corresponds to the **original Needle In A Haystack test** setup.

- **Multi-Needle Retrieval Task (M-RT)**: Explores the LLM's ability to retrieve multiple relevant pieces of information from long texts, simulating complex queries over comprehensive documents.

- **Multi-Needle Reasoning Task (M-RS)**: Assesses LLMs' abilities to integrate multiple key pieces of information extracted from long texts for reasoning, requiring a comprehensive understanding of content.

- **Ancestral Trace Challenge (ATC)**: Tests LLMs' capabilities in handling multi-layer logical challenges within realistic long-text contexts through "kinship trace needles." In the ATC task, no irrelevant (haystack) texts are added; every piece of text is critical, and models must reason through all details for accurate answers.

## Evaluation Steps

> Note: In the latest `OpenCompass` codebase, the NeedleBench dataset is automatically loaded from the [Huggingface interface](https://huggingface.co/datasets/opencompass/NeedleBench), with no need for manual download or configuration.

### `OpenCompass` Environment Setup

```bash
conda create --name opencompass python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate opencompass
git clone https://github.com/open-compass/opencompass opencompass
cd opencompass
pip install -e .
```

### Dataset Configuration

We have pre-configured various long-context settings (4k, 8k, 32k, 128k, 200k, 1000k) in `opencompass/configs/datasets/needlebench`, and you can flexibly define your parameters by adjusting the configuration files.

### Evaluation Example

#### Evaluating with `VLLM` Deployed `Qwen2-5-7B` Model

To evaluate the `Qwen2-5-7B` model deployed with `VLLM` on all tasks under NeedleBench-128K, use the following command. This leverages pre-defined model and dataset configuration files without needing additional configuration:

##### Local Evaluation

If evaluating locally, the command will use all available GPUs. You can control GPU visibility using `CUDA_VISIBLE_DEVICES`:

```bash
# Local evaluation
python run.py --dataset needlebench_128k --models vllm_qwen2_5_7b_instruct_128k  --summarizer needlebench/needlebench_128k_summarizer
```

##### Evaluation on Slurm Cluster

For Slurm environments, you can add options like `--slurm -p partition_name -q reserved --max-num-workers 16`:

```bash
# Slurm evaluation
python run.py --dataset needlebench_128k --models vllm_qwen2_5_7b_instruct_128k --summarizer needlebench/needlebench_128k_summarizer --slurm -p partition_name -q reserved --max-num-workers 16
```

##### Evaluating Specific Subsets

If you only want to test the original Needle In A Haystack task (e.g., single-needle 128k), adjust the dataset parameter:

```bash
python run.py --dataset needlebench_single_128k --models vllm_qwen2_5_7b_instruct_128k --summarizer needlebench/needlebench_128k_summarizer --slurm -p partition_name -q reserved --max-num-workers 16
```

To evaluate only Chinese versions, specify the subset dataset after `/`:

```bash
python run.py --dataset needlebench_single_128k/needlebench_zh_datasets --models vllm_qwen2_5_7b_instruct_128k --summarizer needlebench/needlebench_128k_summarizer --slurm -p partition_name -q reserved --max-num-workers 16
```

Ensure `VLLM` is installed beforehand:

```bash
# Install vLLM with CUDA 12.4.
# For other CUDA versions, please refer to the [official documentation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html)
pip install vllm
```

#### Evaluating Other `Huggingface` Models

For other models, it's recommended to create a custom config file to adjust `max_seq_len` and `max_out_len`, ensuring the model can process the full context. Here is an example (`examples/eval_needlebench.py`):

```python
from mmengine.config import read_base
# we use mmengine.config to import other config files

with read_base():
    from opencompass.configs.models.hf_internlm.hf_internlm2_chat_7b import models as internlm2_chat_7b

    # Evaluate needlebench_32k, adjust the configuration to use 4k, 32k, 128k, 200k, or 1000k if necessary.
    # from .datasets.needlebench.needlebench_32k.needlebench_32k import needlebench_datasets
    # from .summarizers.needlebench import needlebench_32k_summarizer as summarizer

    # only eval original "needle in a haystack test" in needlebench_32k
    from opencompass.configs.datasets.needlebench.needlebench_32k.needlebench_single_32k import needlebench_zh_datasets, needlebench_en_datasets
    from opencompass.configs.summarizers.needlebench import needlebench_32k_summarizer as summarizer

    # eval Ancestral Tracing Challenge(ATC)
    # from .datasets.needlebench.atc.atc_0shot_nocot_2_power_en import needlebench_datasets
    # ATC use default summarizer thus no need to import summarizer

datasets = sum([v for k, v in locals().items() if ('datasets' in k)], [])

for m in internlm2_chat_7b:
    m['max_seq_len'] = 32768 # 保证InternLM2-7B模型能接收到完整的长文本，其他模型需要根据各自支持的最大序列长度修改。
    m['max_out_len'] = 4096

models = internlm2_chat_7b

work_dir = './outputs/needlebench'
```

You can then run evaluation with:

```bash
python run.py configs/eval_needlebench.py --slurm -p partition_name -q reserved --max-num-workers 16
```

No need to manually specify `--dataset`, `--models`, or `--summarizer` again.

### Visualization

NeedleBench's latest version has built-in visualization integrated into the summarizer. You can find corresponding visualizations in the `plots` directory under the output folder without needing additional scripts.

### Citation

If you use NeedleBench, please cite us:

```bibtex
@misc{li2024needlebenchllmsretrievalreasoning,
      title={NeedleBench: Can LLMs Do Retrieval and Reasoning in 1 Million Context Window?},
      author={Mo Li and Songyang Zhang and Yunxin Liu and Kai Chen},
      year={2024},
      eprint={2407.11963},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.11963},
}

@misc{2023opencompass,
    title={OpenCompass: A Universal Evaluation Platform for Foundation Models},
    author={OpenCompass Contributors},
    howpublished={\url{https://github.com/open-compass/opencompass}},
    year={2023}
}

@misc{LLMTest_NeedleInAHaystack,
  title={LLMTest Needle In A Haystack - Pressure Testing LLMs},
  author={gkamradt},
  year={2023},
  howpublished={\url{https://github.com/gkamradt/LLMTest_NeedleInAHaystack}}
}

@misc{wei2023skywork,
      title={Skywork: A More Open Bilingual Foundation Model},
      author={Tianwen Wei and Liang Zhao and Lichang Zhang and Bo Zhu and Lijie Wang and Haihua Yang and Biye Li and Cheng Cheng and Weiwei L\"u and Rui Hu and Chenxia Li and Liu Yang and Xilin Luo and Xuejie Wu and Lunan Liu and Wenjun Cheng and Peng Cheng and Jianhao Zhang and Xiaoyu Zhang and Lei Lin and Xiaokun Wang and Yutuan Ma and Chuanhai Dong and Yanqi Sun and Yifu Chen and Yongyi Peng and Xiaojuan Liang and Shuicheng Yan and Han Fang and Yahui Zhou},
      year={2023},
      eprint={2310.19341},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
