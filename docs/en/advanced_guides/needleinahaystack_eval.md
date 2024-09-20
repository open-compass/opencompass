# Needle In A Haystack Experimental Evaluation

## Introduction to the Needle In A Haystack Test

The Needle In A Haystack test (inspired by [NeedleInAHaystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack/blob/main/LLMNeedleHaystackTester.py)) is an evaluation method that randomly inserts key information into long texts to form prompts for large language models (LLMs). The test aims to detect whether large models can extract such key information from extensive texts, thereby assessing the models' capabilities in processing and understanding long documents.

## Task Overview

Within the `NeedleBench` framework of `OpenCompass`, we have designed a series of increasingly challenging test scenarios to comprehensively evaluate the models' abilities in long text information extraction and reasoning. For a complete introduction, refer to our [technical report](https://arxiv.org/abs/2407.11963):

- **Single-Needle Retrieval Task (S-RT)**: Assesses an LLM's ability to extract a single key piece of information from a long text, testing its precision in recalling specific details within broad narratives. This corresponds to the **original Needle In A Haystack test** setup.

- **Multi-Needle Retrieval Task (M-RT)**: Explores an LLM's capability to retrieve multiple related pieces of information from long texts, simulating real-world scenarios of complex queries on comprehensive documents.

- **Multi-Needle Reasoning Task (M-RS)**: Evaluates an LLM's long-text abilities by extracting and utilizing multiple key pieces of information, requiring the model to have a comprehensive understanding of each key information fragment.

- **Ancestral Trace Challenge (ATC)**: Uses the "relational needle" to test an LLM's ability to handle multi-layer logical challenges in real long texts. In the ATC task, a series of logical reasoning questions are used to test the model's memory and analytical skills for every detail in the text. For this task, we remove the irrelevant text (Haystack) setting, designing all texts as critical information, requiring the LLM to use all the content and reasoning in the text accurately to answer the questions.

### Evaluation Steps

> Note: In the latest code, OpenCompass has been set to automatically load the dataset from [Huggingface API](https://huggingface.co/datasets/opencompass/NeedleBench), so you can **skip directly** the following steps of manually downloading and placing the dataset.

1. Download the dataset from [here](https://github.com/open-compass/opencompass/files/14741330/needlebench.zip).

2. Place the downloaded files in the `opencompass/data/needlebench/` directory. The expected file structure in the `needlebench` directory is shown below:

```
opencompass/
├── configs
├── docs
├── data
│   └── needlebench
│       ├── multi_needle_reasoning_en.json
│       ├── multi_needle_reasoning_zh.json
│       ├── names.json
│       ├── needles.jsonl
│       ├── PaulGrahamEssays.jsonl
│       ├── zh_finance.jsonl
│       ├── zh_game.jsonl
│       ├── zh_government.jsonl
│       ├── zh_movie.jsonl
│       ├── zh_tech.jsonl
│       ├── zh_general.jsonl
├── LICENSE
├── opencompass
├── outputs
├── run.py
├── more...
```

### `OpenCompass` Environment Setup

```bash
conda create --name opencompass python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate opencompass
git clone https://github.com/open-compass/opencompass opencompass
cd opencompass
pip install -e .
```

### Configuring the Dataset

We have pre-configured datasets for common text lengths (4k, 8k, 32k, 128k, 200k, 1000k) in `configs/datasets/needlebench`, allowing you to flexibly create datasets that meet your needs by defining related parameters in the configuration files.

### Evaluation Example

#### Evaluating `InternLM2-7B` Model Deployed Using `LMDeploy`

For example, to evaluate the `InternLM2-7B` model deployed using `LMDeploy` for all tasks in NeedleBench-4K, you can directly use the following command in the command line. This command calls the pre-defined model and dataset configuration files without needing to write additional configuration files:

##### Local Evaluation

If you are evaluating the model locally, the command below will utilize all available GPUs on your machine. You can limit the GPU access for `OpenCompass` by setting the `CUDA_VISIBLE_DEVICES` environment variable. For instance, using `CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py ...` will only expose the first four GPUs to OpenCompass, ensuring that it does not use more than these four GPUs.

```bash
# Local Evaluation
python run.py --dataset needlebench_4k --models lmdeploy_internlm2_chat_7b  --summarizer needlebench/needlebench_4k_summarizer
```

##### Evaluation on a Slurm Cluster

If using `Slurm`, you can add parameters such as `--slurm -p partition_name -q reserved --max-num-workers 16`, as shown below:

```bash
# Slurm Evaluation
python run.py --dataset needlebench_4k --models lmdeploy_internlm2_chat_7b  --summarizer needlebench/needlebench_4k_summarizer --slurm -p partition_name -q reserved --max-num-workers 16
```

##### Evaluating a Subdataset Only

If you only want to test the original NeedleInAHaystack task setup, you could change the dataset parameter to `needlebench_single_4k`, which corresponds to the single needle version of the NeedleInAHaystack test at 4k length:

```bash
python run.py --dataset needlebench_single_4k --models lmdeploy_internlm2_chat_7b  --summarizer needlebench/needlebench_4k_summarizer --slurm -p partition_name -q reserved --max-num-workers 16
```

You can also choose to evaluate a specific subdataset, such as changing the `--datasets` parameter to `needlebench_single_4k/needlebench_zh_datasets` for testing just the Chinese version of the single needle 4K length NeedleInAHaystack task. The parameter after `/` represents the subdataset, which can be found in the dataset variable of `configs/datasets/needlebench/needlebench_4k/needlebench_single_4k.py` :

```bash
python run.py --dataset needlebench_single_4k/needlebench_zh_datasets --models lmdeploy_internlm2_chat_7b  --summarizer needlebench/needlebench_4k_summarizer --slurm -p partition_name -q reserved --max-num-workers 16
```

Be sure to install the [LMDeploy](https://github.com/InternLM/lmdeploy) tool before starting the evaluation:

```bash
pip install lmdeploy
```

This command initiates the evaluation process, with parameters `-p partition_name -q auto` and `--max-num-workers 16` used to specify the Slurm partition name and the maximum number of worker processes.

#### Evaluating Other `Huggingface` Models

For other models, we recommend writing an additional configuration file to modify the model's `max_seq_len` and `max_out_len` parameters so the model can receive the complete long text content, as we have prepared in the `configs/eval_needlebench.py` file. The complete content is as follows:

```python
from mmengine.config import read_base
# We use mmengine.config to import variables from other configuration files

with read_base():
    # from .models.hf_internlm.lmdeploy_internlm2_chat_7b import models as internlm2_chat_7b_200k
    from .models.hf_internlm.hf_internlm2_chat_7b import models as internlm2_chat_7b

    # Evaluate needlebench_4k, adjust the configuration to use 8k, 32k, 128k, 200k, or 1000k if necessary.
    # from .datasets.needlebench.needlebench_4k.needlebench_4k import needlebench_datasets
    # from .summarizers.needlebench import needlebench_4k_summarizer as summarizer

    # only eval original "needle in a haystack test" in needlebench_4k
    from .datasets.needlebench.needlebench_4k.needlebench_single_4k import needlebench_zh_datasets, needlebench_en_datasets
    from .summarizers.needlebench import needlebench_4k_summarizer as summarizer

    # eval Ancestral Tracing Challenge(ATC)
    # from .datasets.needlebench.atc.atc_choice_50 import needlebench_datasets
    # from .summarizers.needlebench import atc_summarizer_50 as summarizer

datasets = sum([v for k, v in locals().items() if ('datasets' in k)], [])

for m in internlm2_chat_7b:
    m['max_seq_len'] = 30768 # Ensure InternLM2-7B model can receive the complete long text, other models need to adjust according to their maximum sequence length support.
    m['max_out_len'] = 2000 # Ensure that in the multi-needle recall task, the model can receive a complete response

models = internlm2_chat_7b

work_dir = './outputs/needlebench'
```

Once the test `config` file is written, we can pass the corresponding config file path through the `run.py` file in the command line, such as:

```bash
python run.py configs/eval_needlebench.py --slurm -p partition_name -q reserved --max-num-workers 16
```

Note, at this point, we do not need to pass in the `--dataset, --models, --summarizer` parameters, as we have already defined these configurations in the config file. You can manually adjust the `--max-num-workers` setting to adjust the number of parallel workers.

### Visualization

We have built-in result visualization into the `summarizer` implementation in the latest code version. You can find the corresponding visualizations in the plots directory of the respective output folder, eliminating the need for manual visualization of scores across various depths and lengths.

If you use this method, please add a reference:

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
      author={Tianwen Wei and Liang Zhao and Lichang Zhang and Bo Zhu and Lijie Wang and Haihua Yang and Biye Li and Cheng Cheng and Weiwei Lü and Rui Hu and Chenxia Li and Liu Yang and Xilin Luo and Xuejie Wu and Lunan Liu and Wenjun Cheng and Peng Cheng and Jianhao Zhang and Xiaoyu Zhang and Lei Lin and Xiaokun Wang and Yutuan Ma and Chuanhai Dong and Yanqi Sun and Yifu Chen and Yongyi Peng and Xiaojuan Liang and Shuicheng Yan and Han Fang and Yahui Zhou},
      year={2023},
      eprint={2310.19341},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

```
