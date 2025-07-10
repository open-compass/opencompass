


# Overview
<p align="center">
   🌐 <a href="https://openstellarteam.github.io/ChineseSimpleQA/" target="_blank">Website</a> • 🤗 <a href="https://huggingface.co/datasets/OpenStellarTeam/Chinese-SimpleQA" target="_blank">Hugging Face</a> • ⏬ <a href="#data" target="_blank">Data</a> •   📃 <a href="https://huggingface.co/datasets/OpenStellarTeam/Chinese-SimpleQA" target="_blank">Paper</a> •   📊 <a href="http://47.109.32.164/" target="_blank">Leaderboard</a>  <br>  <a href="https://github.com/OpenStellarTeam/ChineseSimpleQA/blob/master/README_zh.md">   中文</a> | <a href="https://github.com/OpenStellarTeam/ChineseSimpleQA/blob/master/README.md">English 
</p> 

**Chinese SimpleQA** is the first comprehensive Chinese benchmark to evaluate the factuality ability of language models to answer short questions, and Chinese SimpleQA mainly has five properties (i.e., Chinese, Diverse, High-quality, Static, Easy-to-evaluate). Specifically, our benchmark covers **6 major topics** with **99 diverse subtopics**. 

Please visit our [website](https://openstellarteam.github.io/ChineseSimpleQA/) or check our [paper](https://arxiv.org/abs/2411.07140) for more details. 



## 💫 Instroduction

* How to solve the generative hallucination of models has always been an unsolved problem in the field of artificial intelligence (AI). In order to measure the factual correctness of language models, OpenAI recently released and open-sourced a test set called SimpleQA. We have also been paying attention to the field of  factuality, which currently has problems such as outdated data, inaccurate evaluation, and incomplete coverage. For example, the knowledge evaluation sets widely used now are still CommonSenseQA, CMMLU, and C-Eval, which are multiple-choice question-based evaluation sets. **In order to further promote the research of the Chinese community on the factual correctness of models, we propose the Chinese SimpleQA**.  which consists of 3000 high-quality questions spanning 6 major topics, ranging from humanities to science and engineering. Specifically, the distinct main features of our proposed Chinese SimpleQA dataset are as follows:
  * 🀄**Chinese:** Our Chinese SimpleQA focuses on the Chinese language, which provides a comprehensive evaluation of the factuality abilities of existing LLMs in Chinese.
  * 🍀**Diverse:** Chinese SimpleQA covers 6 topics (i.e., “Chinese Culture”, “Humanities”, “Engineering, Technology, and Applied Sciences”, “Life, Art, and Culture”, “Society”, and “Natural Science”), and these topic includes 99 fine-grained subtopics in total, which demonstrates the diversity of our Chinese SimpleQA. 
  * ⚡**High-quality:** We conduct a comprehensive and rigorous quality control process to ensure the quality and accuracy of our Chinese SimpleQA.
  * 💡**Static:** Following SimpleQA, to preserve the evergreen property of Chinese SimpleQA, all reference answers would not change over time. 
  * 🗂️**Easy-to-evaluate:** Following SimpleQA, as the questions and answers are very short, the grading procedure is fast to run via existing LLMs (e.g., OpenAI API).

- Based on Chinese SimpleQA, we have conducted a comprehensive evaluation of the factual capabilities of existing LLMs. We also maintain a comprehensive leaderboard list. 
- In short, we hope that Chinese SimpleQA can help developers gain a deeper understanding of the factual correctness of their models in the Chinese field, and at the same time provide an important cornerstone for their algorithm research, and jointly promote the growth of Chinese basic models.





## 📊 Leaderboard

详见：  [📊](http://47.109.32.164/)



## ⚖️ Evals

We provide three evaluation methods. 

(1) The first method is based on simple-evals evaluation. The startup command is as follows: 

    ```bash
    python -m simple-evals.demo
    ```
    This will launch evaluations through the OpenAI API.



(2) The second is a simple single evaluation script that we wrote from scratch.  The startup command is as follows: 

- Step1: set your openai key in scripts/chinese_simpleqa_easy.py:

  ```
  os.environ["OPENAI_API_KEY"] = "replace your key here"
  ```

- Step2: run the eval script:

  ```
  python scripts/chinese_simpleqa_easy.py
  ```

- Step3: we also provide a unified processing script for multiple model results. After running it, you can get a complete leaderboard:

  ```
  python scripts/get_leaderboard.py
  ```

  

(3) We also integrated our Chinese SimpleQA benchmark into our forked [OpenCompass](https://github.com/open-compass/opencompass). You can refer to the opencompass configuration script for evaluation
- Step1: git clone Opencompass:
  ```shell
  cd ~
  git clone git@github.com:open-compass/opencompass.git
  cd opencompass
  ```
- Step2: download Chinese Simpleqa data from [huggingface](https://huggingface.co/datasets/OpenStellarTeam/Chinese-SimpleQA),  and put it in the following path(OPENCOMPASS_PATH/data/chinese_simpleqa), make sure you get path like this:
    ```
    ~/opencompass/data/
    └── chinese_simpleqa
        ├── chinese_simpleqa.jsonl
    ```


- Step3: configuration your launch in examples/eval_chinese_simpleqa.py, set your models to be evaluated, set your judge model (we recommend to use gpt4o) and launch it!
  ```
  python run.py examples/eval_chinese_simpleqa.py
  ```


## Citation

Please cite our paper if you use our dataset.

```
@misc{he2024chinesesimpleqachinesefactuality,
      title={Chinese SimpleQA: A Chinese Factuality Evaluation for Large Language Models}, 
      author={Yancheng He and Shilong Li and Jiaheng Liu and Yingshui Tan and Weixun Wang and Hui Huang and Xingyuan Bu and Hangyu Guo and Chengwei Hu and Boren Zheng and Zhuoran Lin and Xuepeng Liu and Dekai Sun and Shirong Lin and Zhicheng Zheng and Xiaoyong Zhu and Wenbo Su and Bo Zheng},
      year={2024},
      eprint={2411.07140},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.07140}, 
}
```

