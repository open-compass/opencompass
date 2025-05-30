
from datasets import load_dataset
from opencompass.datasets.base import BaseDataset
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path
from opencompass.openicl.icl_evaluator import BaseEvaluator
from sklearn.metrics import r2_score,root_mean_squared_error

import os
import numpy as np
import  pandas as pd
import json
import re
import requests
import sympy as sp

@LOAD_DATASET.register_module()
class SRbenchDataset(BaseDataset):
    @staticmethod
    def load(path: str,local_mode=True):
        base_path = get_data_path(path,local_mode=local_mode) 
        formula_csv_path = os.path.join(base_path, f'FeynmanEquation_23.csv')
        data_files_base_dir = os.path.join(base_path, 'Feynman_with_units')
        dataset = load_dataset('csv', data_files=formula_csv_path)['train']
        sample_data=[]
        prompt_1_out=[]
        prompt_2_out=[]
        for row in dataset:
            true_formula = str(row["Formula"])
            n_var=int(row["n_variables"])
            data_filename = str(row['Filename'])

            data_file_path = os.path.join(data_files_base_dir, data_filename)
            full_dataset = np.loadtxt(data_file_path)
            rand_idx = np.random.choice(full_dataset.shape[0], 100, replace=False)
            sampled_data_i = full_dataset[rand_idx]
            if isinstance(sampled_data_i, np.ndarray):
                sample_data.append(sampled_data_i.tolist())
            else:
                sample_data.append(sampled_data_i)
            # x = dataset[:, :n_var]
            # y_true = dataset[:, -1]
            if n_var==2:
                prompt_1 = '\n'.join([f'x0={x1:.4f}, x1={x2:.4f}, y={y:.4f}' for x1, x2, y in sampled_data_i[:-1]])
                prompt_2=f'x0={sampled_data_i[-1, 0]:.4f}, x1={sampled_data_i[-1, 1]:.4f}, y={sampled_data_i[-1, 2]:.4f}'
            else:
                prompt_1 = '\n'.join([f'x0={x1:.4f}, x1={x2:.4f}, x2={x3:.4f},y={y:.4f}' for x1, x2,x3, y in sampled_data_i[:-1]])
                prompt_2=f'x0={sampled_data_i[-1, 0]:.4f}, x1={sampled_data_i[-1, 1]:.4f},x3={sampled_data_i[-1, 2]:.4f}, y={sampled_data_i[-1, 3]:.4f}'


            prompt_1_out.append(prompt_1)
            prompt_2_out.append(prompt_2)
        dataset=dataset.add_column(name="prompt1",column=prompt_1_out)
        dataset=dataset.add_column(name="prompt2",column=prompt_2_out)
        dataset=dataset.add_column(name="data_samples_list",column=sample_data)
        dataset = dataset.rename_column('n_variables', 'n_var')
        return dataset

def mydataset_postprocess(formula_str):
    # 1. 删除 Markdown 残留符号
    formula_str = formula_str.replace('×', '*').replace('·', '*').replace('÷', '/')
    formula_str = formula_str.replace('−', '-').replace('^', '**')
    formula_str = formula_str.replace('“', '"').replace('”', '"').replace('’', "'")

    # 2. 去除 markdown 反引号 ``` 和 $ 符号
    formula_str = formula_str.replace('`', '').replace('$', '').strip()

    # 3. 提取第一行公式（防止有多行解释性输出）
    formula_str = formula_str.split('\n')[0].strip()

    # 4. 用正则去除非合法字符（保留基本数学表达式）
    formula_str = re.sub(r'[^\w\s\+\-\*/\^\=\.\(\)]', '', formula_str)

    # 5. 确保左右去空格
    return formula_str.strip()

class SRbenchDatasetEvaluator(BaseEvaluator):
    def __init__(self,
            local_mode: bool = True,path=""):
            self.dataset=SRbenchDataset.load(path,local_mode=local_mode)
    def parse_formula(self,formula_str, n_var=2):
        try:
            if '=' in formula_str:
                _, expr_str = formula_str.split('=', 1)
            else:
                expr_str = formula_str
            variables = [sp.Symbol(f'x{i}') for i in range(n_var)]
            expr = sp.sympify(expr_str)
            func = sp.lambdify(variables, expr, modules='numpy')
            return func
        except Exception as e:
            print(f'[Parse Error] {formula_str}\n{e}')
            return None
    
    def is_symbolically_equivalent(self,formula1, formula2, n_var=2):
        try:
            x = [sp.Symbol(f'x{i}') for i in range(n_var)]
            expr1 = sp.sympify(formula1.split('=')[1] if '=' in formula1 else formula1)
            expr2 = sp.sympify(formula2.split('=')[1] if '=' in formula2 else formula2)

            return sp.simplify(expr1 - expr2) == 0
        except Exception:
            return False
    
    def score(self, predictions, references) -> dict:
        metrics = {
        'LLM_Score': None,
        'RMSE': None,
        'SymbolicMatch': False,
        'R2': 0}
        metrics_out={
        'LLM_Score': None,
        'RMSE': None,
        'Accuray': False,
        'R2': 0 
        }
        result = pd.DataFrame({
            'GT': pd.Series(dtype=str),
            'Pred': pd.Series(dtype=str),
            'Score': pd.Series(dtype=float),
            'RMSE': pd.Series(dtype=float),
            'R2': pd.Series(dtype=float),
            'SymbolicMatch': pd.Series(dtype=bool)
        })
        # 结构评分（用 LLM）
        for row in range(len(references)):
            #metrics['LLM_Score'] = float(self.llm_evaluate(predictions[row], references[row], mllm='gpt-4o'))
            n_var=self.dataset[row]["n_var"]
            data_sample=self.dataset[row]["data_samples_list"]
            data_sample = np.array(data_sample)
            x=data_sample[:,:n_var]
            y_true=data_sample[:,-1]
            func = self.parse_formula(predictions[row], n_var=n_var)
            if func is not None:
                x_vars = [x[:, i] for i in range(n_var)]
                y_pred = func(*x_vars)
                if np.isscalar(y_pred):
                    y_pred = np.full_like(y_true, y_pred)
                metrics['RMSE'] = root_mean_squared_error(y_true, y_pred)
                metrics['R2'] = r2_score(y_true, y_pred)
            else:
                metrics["R2"]=0
                metrics["RMSE"]= np.inf
            metrics['SymbolicMatch'] = self.is_symbolically_equivalent(predictions[row], references[row], n_var)
            result = result._append({
            'GT': references[row],
            'Pred': predictions[row],
            'RMSE': metrics['RMSE'],
            'R2': metrics['R2'],
            'SymbolicMatch': bool(metrics['SymbolicMatch'])
        }, ignore_index=True)

        if not result.empty:
            symbolic_accuracy = result['SymbolicMatch'].sum() / len(result)
            R2_out = result['R2'].sum() / len(result)
            RMSE_out = result['RMSE'].sum() / len(result)

        metrics_out={
        'RMSE': RMSE_out,
        'R2': R2_out,
        "Accuracy":symbolic_accuracy
        }

        return metrics_out

