# TACO
## Dataset Description
TACO (Topics in Algorithmic COde generation dataset) is a dataset focused on algorithmic code generation, designed to provide a more challenging training dataset and evaluation benchmark for the code generation model field. The dataset consists of programming competition problems that are more difficult and closer to real programming scenarios. It emphasizes improving or evaluating the model's understanding and reasoning abilities in practical application scenarios, rather than just implementing predefined function functionalities.

* Larger scale: TACO includes a training set (25,443 problems) and a test set (1,000 problems), making it the largest code generation dataset currently available.
* Higher quality: Each problem in the TACO dataset is designed to match a diverse set of solution answers, with answer sizes of up to 1.55M. This ensures that the model is not prone to overfitting during training and validates the effectiveness of evaluation results.
* Fine-grained labels: Each problem in the TACO dataset includes fine-grained labels such as task topics, algorithms, skills, and difficulty levels. These labels provide more accurate references for the training and evaluation of code generation models.


## Dataset Structure
```python
DatasetDict({
    train: Dataset({
        features: ['question', 'solutions', 'starter_code', 'input_output', 'difficulty', 'raw_tags', 'name', 'source', 'tags', 'skill_types', 'url', 'Expected Auxiliary Space', 'time_limit', 'date', 'picture_num', 'memory_limit', 'Expected Time Complexity'],
        num_rows: 25443
    })
    test: Dataset({
        features: ['question', 'solutions', 'starter_code', 'input_output', 'difficulty', 'raw_tags', 'name', 'source', 'tags', 'skill_types', 'url', 'Expected Auxiliary Space', 'time_limit', 'date', 'picture_num', 'memory_limit', 'Expected Time Complexity'],
        num_rows: 1000
    })
})
```
## How to Use
You can also specify the difficulties (a list choosing from ["EASY", "MEDIUM", "MEDIUM_HARD", "HARD", "VERY_HARD"] or ["ALL"] as default) or skills (a list choosing from ["Data structures", "Sorting", "Range queries", "Complete search", "Amortized analysis", "Dynamic programming", "Bit manipulation", "Greedy algorithms"] or ["ALL"] as default) by passing the list of difficulties or skills as a list.
```python
from datasets import load_dataset
taco_difficulties = load_dataset('BAAI/TACO', difficulties=['EASY'], token=YOUR_HF_TOKEN)
```
```python
from datasets import load_dataset
taco_skills = load_dataset('BAAI/TACO', skills=['Sorting', 'Range queries'], token=YOUR_HF_TOKEN)
```
## Evaluation results

| dataset              | metric   | CodeLlama-7b-Python | internlm2-chat-1.8b-sft-hf  | internlm2-chat-7b-sft-hf | internlm2-chat-20b-sft-hf |
|-----------------------|----------|-------------|-------------|-------------|-------------|
| TACO                   | pass@1   | 0.7         | 0.7           | 1.7           | 2.7           |


Please refer to [repo](https://github.com/FlagOpen/TACO/tree/main?tab=readme-ov-file) for original results if needed.

## Citation
```
@article{li2023taco,
  title={TACO: Topics in Algorithmic COde generation dataset},
  author={Rongao Li and Jie Fu and Bo-Wen Zhang and Tao Huang and Zhihong Sun and Chen Lyu and Guang Liu and Zhi Jin and Ge Li},
  journal={arXiv preprint arXiv:2312.14852},
  year={2023}
}
```
