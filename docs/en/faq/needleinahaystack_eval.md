# Needle In A Haystack Evaluation

## Introduction to the Needle In A Haystack Test

The Needle In A Haystack test (inspired by [NeedleInAHaystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack/blob/main/LLMNeedleHaystackTester.py)) is an evaluation method where key information is randomly inserted into long texts to form the prompt for large language models (LLMs). This test aims to assess whether LLMs can extract critical information from long texts, thereby evaluating their fundamental ability to comprehend and process long-context documents.

## Task Overview

Within the `OpenCompass` framework, under `NeedleBench`, we designed a series of progressively challenging evaluation tasks to comprehensively assess LLMs' long-text information extraction and reasoning capabilities. For a complete description, please refer to our [technical report](https://arxiv.org/abs/2407.11963).

- **Single-Needle Retrieval Task (S-RT)**: Evaluates the LLM's ability to retrieve a single piece of key information from a long text, testing precise recall of specific details within extensive narratives. This corresponds to the **original Needle In A Haystack test** setup.

- **Multi-Needle Retrieval Task (M-RT)**: Explores the LLM's ability to retrieve multiple relevant pieces of information from long texts, simulating complex queries over comprehensive documents.

- **Multi-Needle Reasoning Task (M-RS)**: Assesses LLMs' abilities to integrate multiple key pieces of information extracted from long texts for reasoning, requiring a comprehensive understanding of content.

- **Ancestral Trace Challenge (ATC)**: Tests LLMs' capabilities in handling multi-layer logical challenges within realistic long-text contexts through "kinship trace needles." In the ATC task, no irrelevant (haystack) texts are added; every piece of text is critical, and models must reason through all details for accurate answers.

> **Note:** NeedleBench (v2) includes several optimizations and adjustments in dataset construction and task details. For a detailed comparison between the old and new versions, as well as a summary of updates, please refer to [opencompass/configs/datasets/needlebench_v2/readme.md](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/needlebench_v2/readme.md).

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

We have pre-configured various long-context settings (4k, 8k, 32k, 128k, 200k, 1000k) in `opencompass/configs/datasets/needlebench_v2`, and you can flexibly define your parameters by adjusting the configuration files.

### Evaluation Example

#### Evaluating with `VLLM` Deployed `Qwen2-5-7B` Model

To evaluate the `Qwen2-5-7B` model deployed with `VLLM` on all tasks under NeedleBench-128K, use the following command. This leverages pre-defined model and dataset configuration files without needing additional configuration:

##### Local Evaluation

If evaluating locally, the command will use all available GPUs. You can control GPU visibility using `CUDA_VISIBLE_DEVICES`:

```bash
# Local evaluation
python run.py --datasets needlebench_v2_128k --models vllm_qwen2_5_7b_instruct_128k  --summarizer needlebench/needlebench_v2_128k_summarizer
```

##### Evaluation on Slurm Cluster

For Slurm environments, you can add options like `--slurm -p partition_name -q reserved --max-num-workers 16`:

```bash
# Slurm evaluation
python run.py --datasets needlebench_v2_128k --models vllm_qwen2_5_7b_instruct_128k --summarizer needlebench/needlebench_v2_128k_summarizer --slurm -p partition_name -q reserved --max-num-workers 16
```

##### Evaluating Specific Subsets

If you only want to test the original Needle In A Haystack task (e.g., single-needle 128k), adjust the dataset parameter:

```bash
python run.py --datasets needlebench_v2_single_128k --models vllm_qwen2_5_7b_instruct_128k --summarizer needlebench/needlebench_v2_128k_summarizer --slurm -p partition_name -q reserved --max-num-workers 16
```

To evaluate only Chinese versions, specify the subset dataset after `/`:

```bash
python run.py --datasets needlebench_v2_single_128k/needlebench_zh_datasets --models vllm_qwen2_5_7b_instruct_128k --summarizer needlebench/needlebench_v2_128k_summarizer --slurm -p partition_name -q reserved --max-num-workers 16
```

Ensure `VLLM` is installed beforehand:

```bash
# Install vLLM with CUDA 12.4.
# For other CUDA versions, please refer to the [official documentation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html)
pip install vllm
```

#### Evaluating Other `Huggingface` Models

For other models, it is recommended to write your own config file (such as `examples/eval_needlebench_v2.py`) to adjust `max_seq_len` and `max_out_len`, so that the model can process the full context.

You can then run evaluation with:

```bash
python run.py examples/eval_needlebench_v2.py --slurm -p partition_name -q reserved --max-num-workers 16
```

No need to manually specify `--datasets`, `--models`, or `--summarizer` again.

### Visualization

NeedleBench's latest version has built-in visualization integrated into the summarizer. You can find corresponding visualizations in the `plots` directory under the output folder without needing additional scripts.

### Citation

If you use NeedleBench, please cite us:

```bibtex
@misc{li2025needlebenchllmsretrievalreasoning,
      title={NeedleBench: Can LLMs Do Retrieval and Reasoning in Information-Dense Context?}, 
      author={Mo Li and Songyang Zhang and Taolin Zhang and Haodong Duan and Yunxin Liu and Kai Chen},
      year={2025},
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
