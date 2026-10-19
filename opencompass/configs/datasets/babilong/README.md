# BABILong
OpenCompass now supports the brand new long-context language model evaluation benchmark — [BABILong](https://arxiv.org/pdf/2406.10149). BABILong provides an evaluation of long-context reasoning across extremely long documents, including a diverse set of 20 reasoning tasks such as fact chaining, simple induction, deduction, counting, and handling lists/sets. This benchmark is designed to test the ability of language models to reason over facts distributed in long natural text, and it allows for the construction of tasks of almost arbitrary length to adapt to the evaluation of new, more powerful models in an extensible and controllable way.



## How to Use
The BABILong dataset is available on Hugging Face: [RMT-team/babilong](https://huggingface.co/datasets/RMT-team/babilong). Opencompass provides an automatic download for BABILong dataset, due to the dataset size, we only provide the data up to 1M tokens. For longer context, you can download the dataset from Hugging Face directly.

BABILong paper provides in total 20 tasks, we provide 10 tasks configurations in OpenCompass and they are organized by different context sizes. You can create your own configurations by following the examples in `opencompass/configs/datasets/babilong/babilong_1m_gen.py`.

Opencompass provides a demo for evaluating language models on the BABILong dataset.

```bash
opencompass examples/eval_babilong.py
```
OpenCompass provides the results of some models on the BABILong dataset. The evaluation results are run with LMDeploy with default model settings.

| dataset | version | metric | mode | internlm2_5-7b-chat-turbomind | qwen2.5-7b-instruct-turbomind | llama-3_1-8b-instruct-turbomind | ministral-8B-instruct-2410-turbomind |
|----- | ----- | ----- | ----- | ----- | ----- | ----- | -----|
| babilong_0k | - | naive_average | gen | 76.51 | 80.25 | 76.44 | 76.40 |
| babilong_4k | - | naive_average | gen | 67.55 | 70.35 | 67.41 | 67.92 |
| babilong_16k | - | naive_average | gen | 53.78 | 65.83 | 60.26 | 56.58 |
| babilong_32k | - | naive_average | gen | 50.86 | 62.66 | 59.56 | 53.52 |
| babilong_128k | - | naive_average | gen | 39.33 | 27.79 | 52.01 | 3.20 |
| babilong_256k | - | naive_average | gen | 17.31 | 7.30 | 23.35 | 9.50 |

## Citation

```bibtex
@misc{kuratov2024babilong,
    title={BABILong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack}, 
    author={Yuri Kuratov and Aydar Bulatov and Petr Anokhin and Ivan Rodkin and Dmitry Sorokin and Artyom Sorokin and Mikhail Burtsev},
    year={2024},
    eprint={2406.10149},
    archivePrefix={arXiv}
}
```