# MMLU-CF: A Contamination-free Multi-task Language Understanding Benchmark

<div align="center">

![](https://img.shields.io/badge/Task-MMLU_CF-orange)
![](https://img.shields.io/badge/Data-Released-green)
![](https://img.shields.io/badge/Code_License-MIT-blue)

</div>

<p align="center">
  <a href="https://arxiv.org/pdf/2412.15194"><b>[📜 Paper]</b></a> •
  <a href="https://huggingface.co/datasets/microsoft/MMLU-CF"><b>[🤗 HF Dataset]</b></a> •
  <a href="https://github.com/microsoft/MMLU-CF"><b>[🐱 GitHub]</b></a>
</p>

## 📢 News and Updates
[2024.12.01] 🔥We have initialized the repository.  
[2024.12.16] 🔥We have added the evaluation results of Phi-4-14B and Llama-3.3-70B-Instruct.  
[2024.12.20] 🔥We have released the validation dataset of MMLU-CF.  


## 1. The Motivation of MMLU-CF

<!-- MMLU-CF is a contamination-free and more challenging multiple-choice question benchmark. This dataset contains 10K questions each for the validation set and test set, covering various disciplines. -->

- The open-source nature of these benchmarks and the broad sources of training data for LLMs have inevitably led to benchmark contamination, resulting in unreliable evaluation results. To alleviate this issue, we propose MMLU-CF.
- (a) An instance of leakage in MMLU. When questions are used as prompt from the MMLU, certain LLMs, due to their memorization capabilities, directly provide **choices identical to the original ones**. (b) When questions are used as prompt from the MMLU-CF, LLMs only provide guessed choices.
This indicates that the MMLU test set suffers from data contamination and memorization by some LLMs, while the proposed MMLU-CF avoids such leakage.
<p float="center">
  <img src="./Figures/Fig_1_a.png" alt="Fig1_a" width="45%" />
  <img src="./Figures/Fig_1_b.png" alt="Fig1_b" width="45%" />
</p>


## 2. How to Evaluate Your Models on the MMLU-CF Validation/Test Set

  #### (1) We perform automated testing only on Huggingface models. After following the steps outlined below and obtaining the validation set results from [OpenCompass](https://github.com/open-compass/opencompass), the test set results can then be accessed via GitHub Issues. 
  
  **Step 1**. **Validation set evaluation**: Obtaining the validation results for your model using LLM evaluation tools, [OpenCompass](https://github.com/open-compass/opencompass). The validation dataset download from [🤗 Huggingface](https://huggingface.co/datasets/microsoft/MMLU-CF). The data directory structure in the opencompass:

```
data
└── mmlu_cf 
    ├── dev
    └── val
```
  
  **Step 2**. **Test set evaluation**: With the validation results, submit a GitHub issue on the [MMLU-CF](https://github.com/) GitHub homepage to request the test set results. Please follow the format below:

Example 1,
```
Title: 
Test set evaluation Request - add HF model [microsoft/phi-4]  
Content: 
The result on validation set: 68.5%
```
Example 2,
<p>
  <img src="./Figures/Fig_6.png" alt="Fig6" width="80%" style="display: block; margin: 0 auto;" />
</p>

  **Notably**:
   - Ensure you use the format with square brackets `[ ]` as shown. The model name **microsoft/phi-4** corresponds to the name on HuggingFace.
   - We will automatically submit your model. The time to receive the results depends on the number of models being evaluated, but it typically takes **1-2 weeks**.

  #### (2) For API models, if OpenCompass updates the model interface, you can obtain the test set results by sending a temporary key to [Email](yangyu.huang@microsoft.com) after receiving the validation set results.


## 3. What is the Difference between MMLU-CF and MMLU
MMLU focuses on the breadth and reasoning without considering contamination prevention. We apply three decontamination rules to mitigate unintentional data leakage while collecting data from a broader domain. Meanwhile, our MMLU-CF benchmark maintains the test set closed-source to prevent malicious data leakage.

<p float="left">
  <img src="./Figures/Fig_4.png" alt="Fig4" width="55%" />
  <span style="display:inline-block; width: 10%;"></span>
  <img src="./Figures/Fig_5.png" alt="Fig5" width="30%" />
</p>


## 4. Leaderboard
<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="1">MMLU </th>
      <th colspan="6">MMLU-CF </th>
    </tr>
    <tr>
      <th>5-shot   </th>
      <th>5-shot Test   </th>
      <th>5-shot Validation  </th>
      <th>5-shot Δ   </th>
      <th>0-shot Test   </th>
      <th>0-shot Validation    </th>
      <th>0-shot Δ    </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>API</strong></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>GPT-4o</td>
      <td>88.0</td>
      <td>73.4</td>
      <td>73.4</td>
      <td>+0.0</td>
      <td>71.9</td>
      <td>72.4</td>
      <td>-0.5</td>
    </tr>
    <tr>
      <td>GPT-4-Turbo</td>
      <td>86.5</td>
      <td>70.4</td>
      <td>70.1</td>
      <td>+0.3</td>
      <td>68.9</td>
      <td>68.7</td>
      <td>+0.1</td>
    </tr>
    <tr>
      <td>GPT-4o-mini</td>
      <td>81.8</td>
      <td>65.5</td>
      <td>65.1</td>
      <td>+0.4</td>
      <td>66.0</td>
      <td>65.3</td>
      <td>+0.7</td>
    </tr>
    <tr>
      <td>Gemini-1.5-Flash</td>
      <td>78.7</td>
      <td>64.8</td>
      <td>64.9</td>
      <td>-0.1</td>
      <td>56.7</td>
      <td>56.9</td>
      <td>-0.2</td>
    </tr>
    <tr>
      <td>GPT-3.5-Turbo</td>
      <td>71.4</td>
      <td>58.2</td>
      <td>59.0</td>
      <td>-0.8</td>
      <td>57.2</td>
      <td>58.1</td>
      <td>-0.9</td>
    </tr>
    <tr>
      <td><strong>Large</strong></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Qwen2.5-72B-instruct</td>
      <td>85.3</td>
      <td>71.6</td>
      <td>71.3</td>
      <td>+0.3</td>
      <td>70.6</td>
      <td>70.4</td>
      <td>+0.2</td>
    </tr>
    <tr>
      <td>Llama-3-70B-instruct</td>
      <td>82.0</td>
      <td>68.9</td>
      <td>68.8</td>
      <td>+0.1</td>
      <td>68.1</td>
      <td>67.4</td>
      <td>+0.7</td>
    </tr>
    <tr>
      <td>Llama-3.3-70B-instruct</td>
      <td>86.3</td>
      <td>68.8</td>
      <td>67.8</td>
      <td>+1.0</td>
      <td>67.6</td>
      <td>67.5</td>
      <td>+0.1</td>
    </tr>
    <tr>
      <td>Llama-3.1-70B-instruct</td>
      <td>86.0</td>
      <td>68.7</td>
      <td>68.1</td>
      <td>+0.6</td>
      <td>70.4</td>
      <td>69.7</td>
      <td>+0.7</td>
    </tr>
    <tr>
      <td>Phi-3.5-MoE-instruct</td>
      <td>78.9</td>
      <td>64.6</td>
      <td>64.5</td>
      <td>+0.1</td>
      <td>63.1</td>
      <td>62.1</td>
      <td>+1.0</td>
    </tr>
    <tr>
      <td>Qwen2-72B-instruct</td>
      <td>82.3</td>
      <td>63.7</td>
      <td>64.3</td>
      <td>-0.6</td>
      <td>62.4</td>
      <td>62.5</td>
      <td>-0.1</td>
    </tr>
    <tr>
      <td>Mixtral-8x22B-instruct</td>
      <td>76.2</td>
      <td>62.8</td>
      <td>62.5</td>
      <td>+0.3</td>
      <td>65.3</td>
      <td>64.8</td>
      <td>+0.5</td>
    </tr>
    <tr>
  <td>Qwen1.5-72B-chat</td>
  <td>75.6</td>
  <td>59.8</td>
  <td>60.2</td>
  <td>-0.4</td>
  <td>59.1</td>
  <td>59.6</td>
  <td>-0.5</td>
</tr>
<tr>
  <td>Llama-2-70B-chat</td>
  <td>68.9</td>
  <td>52.2</td>
  <td>51.8</td>
  <td>+0.4</td>
  <td>51.2</td>
  <td>50.9</td>
  <td>+0.3</td>
</tr>
<tr>
  <td><strong>Medium</strong></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
</tr>
<tr>
  <td>Qwen2.5-32B-instruct</td>
  <td>83.9</td>
  <td>69.7</td>
  <td>68.8</td>
  <td>+0.9</td>
  <td>68.9</td>
  <td>68.8</td>
  <td>+0.1</td>
</tr>
<tr>
  <td>Phi-4-14B</td>
  <td>84.8</td>
  <td>67.8</td>
  <td>68.5</td>
  <td>-0.7</td>
  <td>68.5</td>
  <td>69.4</td>
  <td>-0.9</td>
</tr>
<tr>
  <td>Qwen2.5-14B-instruct</td>
  <td>79.9</td>
  <td>66.4</td>
  <td>66.1</td>
  <td>+0.3</td>
  <td>67.0</td>
  <td>66.0</td>
  <td>+1.0</td>
</tr>
<tr>
  <td>Phi-3-medium-instruct</td>
  <td>77.9</td>
  <td>64.2</td>
  <td>64.2</td>
  <td>+0.0</td>
  <td>62.5</td>
  <td>62.7</td>
  <td>-0.2</td>
</tr>
<tr>
  <td>Gemma2-27B</td>
  <td>75.2</td>
  <td>63.9</td>
  <td>63.5</td>
  <td>+0.4</td>
  <td>64.2</td>
  <td>64.0</td>
  <td>+0.2</td>
</tr>
<tr>
  <td>Yi-1.5-34B-chat</td>
  <td>76.8</td>
  <td>61.3</td>
  <td>60.5</td>
  <td>+0.8</td>
  <td>60.6</td>
  <td>59.5</td>
  <td>+1.1</td>
</tr>
<tr>
  <td>Mixtral-8x7B-instruct-v0.1</td>
  <td>70.5</td>
  <td>58.3</td>
  <td>57.1</td>
  <td>-1.2</td>
  <td>58.9</td>
  <td>58.5</td>
  <td>+0.4</td>
</tr>
<tr>
  <td>Deepseek-v2-lite-chat</td>
  <td>55.7</td>
  <td>49.3</td>
  <td>48.7</td>
  <td>+0.6</td>
  <td>48.2</td>
  <td>47.7</td>
  <td>+0.5</td>
</tr>
<tr>
  <td>Baichuan-2-13B-chat</td>
  <td>57.3</td>
  <td>48.3</td>
  <td>48.6</td>
  <td>-0.3</td>
  <td>47.1</td>
  <td>48.1</td>
  <td>-1.0</td>
</tr>
<tr>
  <td>Llama-2-13B-chat</td>
  <td>54.8</td>
  <td>42.8</td>
  <td>42.1</td>
  <td>+0.7</td>
  <td>44.8</td>
  <td>44.6</td>
  <td>+0.2</td>
</tr>
<tr>
  <td><strong>Small</strong></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
</tr>
<tr>
  <td>Qwen2.5-7B-instruct</td>
  <td>75.4</td>
  <td>61.3</td>
  <td>60.4</td>
  <td>+0.9</td>
  <td>59.3</td>
  <td>58.6</td>
  <td>+0.7</td>
</tr>
<tr>
  <td>Qwen2-7B-instruct</td>
  <td>70.5</td>
  <td>58.1</td>
  <td>57.9</td>
  <td>+0.2</td>
  <td>58.3</td>
  <td>57.4</td>
  <td>+0.9</td>
</tr>
<tr>
  <td>Glm-4-9B-chat</td>
  <td>72.4</td>
  <td>57.8</td>
  <td>57.9</td>
  <td>-0.1</td>
  <td>58.6</td>
  <td>58.7</td>
  <td>-0.1</td>
</tr>
<tr>
  <td>Internlm-2.5-7B-chat</td>
  <td>72.8</td>
  <td>57.3</td>
  <td>56.8</td>
  <td>+0.5</td>
  <td>57.9</td>
  <td>56.9</td>
  <td>+1.0</td>
</tr>
<tr>
  <td>Llama-3-8B-instruct</td>
  <td>68.4</td>
  <td>57.3</td>
  <td>56.5</td>
  <td>+0.8</td>
  <td>56.4</td>
  <td>55.4</td>
  <td>+1.0</td>
</tr>
<tr>
  <td>Llama-3.1-8B-instruct</td>
  <td>68.1</td>
  <td>57.1</td>
  <td>57.9</td>
  <td>-0.8</td>
  <td>56.1</td>
  <td>56.1</td>
  <td>+0.0</td>
</tr>
<tr>
  <td>Gemma-2-9B</td>
  <td>71.3</td>
  <td>53.7</td>
  <td>53.3</td>
  <td>+0.4</td>
  <td>32.1</td>
  <td>31.2</td>
  <td>+0.9</td>
</tr>
<tr>
  <td>Yi-1.5-6B-chat</td>
  <td>62.8</td>
  <td>52.8</td>
  <td>51.4</td>
  <td>+1.4</td>
  <td>52.2</td>
  <td>51.9</td>
  <td>+0.3</td>
</tr>
<tr>
  <td>Mistral-7B-instruct-v0.3</td>
  <td>60.3</td>
  <td>50.7</td>
  <td>50.9</td>
  <td>-0.2</td>
  <td>51.1</td>
  <td>50.9</td>
  <td>+0.2</td>
</tr>
<tr>
  <td>Baichuan-2-7B-chat</td>
  <td>52.9</td>
  <td>44.5</td>
  <td>43.9</td>
  <td>+0.6</td>
  <td>43.9</td>
  <td>44.0</td>
  <td>-0.1</td>
</tr>
<tr>
  <td>Llama-2-7B-chat</td>
  <td>45.3</td>
  <td>39.4</td>
  <td>38.5</td>
  <td>+0.9</td>
  <td>41.9</td>
  <td>40.9</td>
  <td>+1.0</td>
</tr>
<tr>
  <td><strong>Mini</strong></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
</tr>
<tr>
  <td>Phi-3-mini-instruct (3.8B)</td>
  <td>70.9</td>
  <td>57.9</td>
  <td>58.1</td>
  <td>-0.2</td>
  <td>58.2</td>
  <td>57.5</td>
  <td>+0.7</td>
</tr>
<tr>
  <td>Phi-3.5-mini-instruct (3.8B)</td>
  <td>69.1</td>
  <td>57.9</td>
  <td>57.4</td>
  <td>+0.5</td>
  <td>58.3</td>
  <td>57.7</td>
  <td>+0.6</td>
</tr>
<tr>
  <td>Qwen2.5-3B-instruct</td>
  <td>64.4</td>
  <td>55.9</td>
  <td>56.4</td>
  <td>-0.5</td>
  <td>54.3</td>
  <td>53.9</td>
  <td>+0.4</td>
</tr>
<tr>
  <td>Qwen2.5-1.5B-instruct</td>
  <td>50.7</td>
  <td>51.2</td>
  <td>51.0</td>
  <td>+0.2</td>
  <td>50.7</td>
  <td>50.4</td>
  <td>+0.3</td>
</tr>
<tr>
  <td>Qwen2-1.5B-instruct</td>
  <td>52.4</td>
  <td>47.1</td>
  <td>47.5</td>
  <td>-0.4</td>
  <td>45.2</td>
  <td>44.5</td>
  <td>+0.7</td>
</tr>
<tr>
  <td>Gemma-2-2B</td>
  <td>51.3</td>
  <td>43.9</td>
  <td>42.4</td>
  <td>+1.5</td>
  <td>30.5</td>
  <td>29.4</td>
  <td>+0.9</td>
</tr>
<tr>
  <td>Qwen2.5-0.5B-instruct</td>
  <td>24.1</td>
  <td>41.9</td>
  <td>41.1</td>
  <td>+0.8</td>
  <td>36.0</td>
  <td>34.9</td>
  <td>+1.1</td>
</tr>
<tr>
  <td>Internlm-2-chat-1.8b</td>
  <td>47.1</td>
  <td>40.5</td>
  <td>39.4</td>
  <td>+1.1</td>
  <td>41.2</td>
  <td>39.8</td>
  <td>+1.4</td>
</tr>
<tr>
  <td>Qwen2-0.5B-instruct</td>
  <td>37.9</td>
  <td>38.3</td>
  <td>38.3</td>
  <td>+0.0</td>
  <td>33.5</td>
  <td>33.5</td>
  <td>+0.0</td>
</tr>
  </tbody>
</table>

## 5. Data Construction Pipeline
![Fig3](./Figures/Fig_3.png)
The pipeline involves (1) MCQ Collection to gather a diverse set of questions; (2) MCQ Cleaning to ensure quality; (3) Difficulty Sampling to ensure an appropriate difficulty distribution for questions; (4) LLMs checking: The LLMs, including GPT-4o, Gemini, and Claude, are reviewing the accuracy and safety of the data; and (5) Contamination-Free Processing to prevent data leakage and maintain dataset purity. Ultimately, this process results in the MMLU-CF, consisting of 10,000 questions for the closed-source test set and 10,000 for the open-source validation set.

## 6. Contact
For any inquiries or concerns, feel free to reach out to us via Email: [Qihao Zhao](qhzhaoo@gmail.com) and [Yangyu Huang](yanghuan@microsoft.com).

## 7. Citation
```
@misc{zhao2024mmlucfcontaminationfreemultitasklanguage,
      title={MMLU-CF: A Contamination-free Multi-task Language Understanding Benchmark}, 
      author={Qihao Zhao and Yangyu Huang and Tengchao Lv and Lei Cui and Qinzheng Sun and Shaoguang Mao and Xin Zhang and Ying Xin and Qiufeng Yin and Scarlett Li and Furu Wei},
      year={2024},
      eprint={2412.15194},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.15194}, 
}
```

## 8. License
This repository is licensed under the [MIT](https://github.com/microsoft/PEACE/blob/main/LICENSE) License.
The validation dataset of MMLU-CF is subject to the [CDLA-2.0](https://cdla.dev/permissive-2-0/) License.
