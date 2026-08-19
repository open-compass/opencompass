# Guideline for evaluating HelloBench on Diverse LLMs

HelloBench is a comprehensive, in-the-wild, and open-ended benchmark to evaluate LLMs' performance in generating long text. More details could be found in [🌐Github Repo](https://github.com/Quehry/HelloBench) and [📖Paper](https://arxiv.org/abs/2409.16191).

## Detailed instructions to evaluate HelloBench in Opencompass

1. Git clone Opencompass

```shell
cd ~
git clone git@github.com:open-compass/opencompass.git
cd opencompass
```

2. Download HelloBench data in [Google Drive Url](https://drive.google.com/file/d/1EJTmMFgCs2pDy9l0wB5idvp3XzjYEsi9/view?usp=sharing), unzip it and put it in the following path(OPENCOMPASS_PATH/data/HelloBench), make sure you get path like this:

```
~/opencompass/data/
└── HelloBench
    ├── chat.jsonl
    ├── heuristic_text_generation.jsonl
    ├── length_constrained_data
    │   ├── heuristic_text_generation_16k.jsonl
    │   ├── heuristic_text_generation_2k.jsonl
    │   ├── heuristic_text_generation_4k.jsonl
    │   └── heuristic_text_generation_8k.jsonl
    ├── open_ended_qa.jsonl
    ├── summarization.jsonl
    └── text_completion.jsonl
```

3. Setup your opencompass

```
cd ~/opencompass
pip install -e .
```

4. configuration your launch in examples/eval_hellobench.py

- set your models to be evaluated

- set your judge model (we recommend to use gpt4o-mini)

5. launch it!

```
python run.py examples/eval_hellobench.py
```

6. After that, you could find the results in outputs/hellobench/xxx/summary
