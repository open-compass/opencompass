# APPS
## Dataset Description
APPS is a benchmark for code generation with 10000 problems. It can be used to evaluate the ability of language models to generate code from natural language specifications.

## Dataset Structure
```python
DatasetDict({
    train: Dataset({
        features: ['problem_id', 'question', 'solutions', 'input_output', 'difficulty', 'url', 'starter_code'],
        num_rows: 5000
    })
    test: Dataset({
        features: ['problem_id', 'question', 'solutions', 'input_output', 'difficulty', 'url', 'starter_code'],
        num_rows: 5000
    })
})
```
## How to Use
You can also filter the dataset based on difficulty level: introductory, interview and competition. Just pass a list of difficulty levels to the filter. For example, if you want the most challenging questions, you need to select the competition level:
```python
ds = load_dataset("codeparrot/apps", split="train", difficulties=["competition"])
print(next(iter(ds))["question"])
```
## Evaluation results


| Dataset | Metric | Baichuan2-7B | Baichuan2-13B | InternLM2-7B | InternLM2-20B |
|---------|--------|---------------|----------------|---------------|----------------|
| APPS(testset) | pass@1 | 0.0 | 0.06 | 0.0 | 0.0 |

Please refer to Table 3 of [code llama](https://scontent-nrt1-2.xx.fbcdn.net/v/t39.2365-6/369856151_1754812304950972_1159666448927483931_n.pdf?_nc_cat=107&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=TxT1PKkNBZoAX8zMHbm&_nc_ht=scontent-nrt1-2.xx&oh=00_AfDmmQAPzqX1-QOKIDUV5lGKzaZqt0CZUVtxFjHtnh6ycQ&oe=65F5AF8F) for original results if needed. 

## Citation
```
@article{hendrycksapps2021,
  title={Measuring Coding Challenge Competence With APPS},
  author={Dan Hendrycks and Steven Basart and Saurav Kadavath and Mantas Mazeika and Akul Arora and Ethan Guo and Collin Burns and Samir Puranik and Horace He and Dawn Song and Jacob Steinhardt},
  journal={NeurIPS},
  year={2021}
}
```