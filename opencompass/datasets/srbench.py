# flake8: noqa
import os
import re

import numpy as np
import pandas as pd
import sympy as sp
from datasets import load_dataset
from sklearn.metrics import r2_score, root_mean_squared_error

from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path


@LOAD_DATASET.register_module()
class SRbenchDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        path = get_data_path(path)
        base_path = os.path.join(path, 'Feynman')
        formula_csv_path = os.path.join(base_path, 'FeynmanEquation_23.csv')
        data_files_base_dir = os.path.join(base_path, 'Feynman_with_units')
        dataset = load_dataset('csv', data_files=formula_csv_path)['train']
        sample_data = []
        prompt_1_out = []
        prompt_2_out = []
        for row in dataset:
            n_var = int(row['n_variables'])
            data_filename = str(row['Filename'])

            data_file_path = os.path.join(data_files_base_dir, data_filename)
            full_dataset = np.loadtxt(data_file_path)
            rand_idx = np.random.choice(full_dataset.shape[0],
                                        100,
                                        replace=False)
            sampled_data_i = full_dataset[rand_idx]
            if isinstance(sampled_data_i, np.ndarray):
                sample_data.append(sampled_data_i.tolist())
            else:
                sample_data.append(sampled_data_i)
            # x = dataset[:, :n_var]
            # y_true = dataset[:, -1]
            if n_var == 2:
                prompt_1 = '\n'.join([
                    f'x0={x1:.4f}, x1={x2:.4f}, y={y:.4f}'
                    for x1, x2, y in sampled_data_i[:-1]
                ])
                prompt_2 = f'x0={sampled_data_i[-1, 0]:.4f}, x1={sampled_data_i[-1, 1]:.4f}, y={sampled_data_i[-1, 2]:.4f}'
            else:
                prompt_1 = '\n'.join([
                    f'x0={x1:.4f}, x1={x2:.4f}, x2={x3:.4f},y={y:.4f}'
                    for x1, x2, x3, y in sampled_data_i[:-1]
                ])
                prompt_2 = f'x0={sampled_data_i[-1, 0]:.4f}, x1={sampled_data_i[-1, 1]:.4f},x3={sampled_data_i[-1, 2]:.4f}, y={sampled_data_i[-1, 3]:.4f}'

            prompt_1_out.append(prompt_1)
            prompt_2_out.append(prompt_2)
        dataset = dataset.add_column(name='prompt1', column=prompt_1_out)
        dataset = dataset.add_column(name='prompt2', column=prompt_2_out)
        dataset = dataset.add_column(name='data_samples_list',
                                     column=sample_data)
        dataset = dataset.rename_column('n_variables', 'n_var')
        return dataset


def mydataset_postprocess(formula_str):

    formula_str = formula_str.replace('×', '*').replace('·',
                                                        '*').replace('÷', '/')
    formula_str = formula_str.replace('−', '-').replace('^', '**')
    formula_str = formula_str.replace('\“',
                                      '"').replace('\”',
                                                   '"').replace('’', "'")

    formula_str = formula_str.replace('`', '').replace('$', '').strip()

    formula_str = formula_str.split('\n')[0].strip()

    formula_str = re.sub(r'[^\w\s\+\-\*/\^\=\.\(\)]', '', formula_str)

    return formula_str.strip()


class SRbenchDatasetEvaluator(BaseEvaluator):

    def __init__(self, path=''):
        self.dataset = SRbenchDataset.load(path)

    def parse_formula(self, formula_str, n_var=2):
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

    def is_symbolically_equivalent(self, formula1, formula2, n_var=2):
        try:
            expr1 = sp.sympify(
                formula1.split('=')[1] if '=' in formula1 else formula1)
            expr2 = sp.sympify(
                formula2.split('=')[1] if '=' in formula2 else formula2)

            return sp.simplify(expr1 - expr2) == 0
        except Exception:
            return False

    def score(self, predictions, references) -> dict:
        metrics = {
            'LLM_Score': None,
            'RMSE': None,
            'SymbolicMatch': False,
            'R2': 0
        }
        metrics_out = {
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

        for row in range(len(references)):
            # metrics['LLM_Score'] = float(self.llm_evaluate(predictions[row], references[row], mllm='gpt-4o'))
            n_var = self.dataset[row]['n_var']
            data_sample = self.dataset[row]['data_samples_list']
            data_sample = np.array(data_sample)
            x = data_sample[:, :n_var]
            y_true = data_sample[:, -1]
            func = self.parse_formula(predictions[row], n_var=n_var)
            if func is not None:
                try:
                    x_vars = [x[:, i] for i in range(n_var)]
                    y_pred = func(*x_vars)
                    # 确保y_pred是数值类型
                    if hasattr(y_pred, 'is_number') and not y_pred.is_number:
                        raise TypeError('Expression is not a number')
                    if np.isscalar(y_pred):
                        y_pred = np.full_like(y_true, y_pred)
                    # 过滤掉 NaN
                    y_true = np.array(y_true, dtype=np.float64)
                    y_pred = np.array(y_pred, dtype=np.float64)
                    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
                    if np.any(mask):
                        metrics['RMSE'] = root_mean_squared_error(
                            y_true[mask], y_pred[mask])
                        metrics['R2'] = r2_score(y_true[mask], y_pred[mask])
                    else:
                        metrics['RMSE'] = np.inf
                        metrics['R2'] = 0
                except (TypeError, ValueError, ZeroDivisionError) as e:
                    print(f'Error evaluating formula: {e}')
                    metrics['RMSE'] = np.inf
                    metrics['R2'] = 0
            else:
                metrics['R2'] = 0
                metrics['RMSE'] = np.inf
            metrics['SymbolicMatch'] = self.is_symbolically_equivalent(
                predictions[row], references[row], n_var)
            result = result._append(
                {
                    'GT': references[row],
                    'Pred': predictions[row],
                    'RMSE': metrics['RMSE'],
                    'R2': metrics['R2'],
                    'SymbolicMatch': bool(metrics['SymbolicMatch'])
                },
                ignore_index=True)

        if not result.empty:
            symbolic_accuracy = result['SymbolicMatch'].sum() / len(result)
            # 排除inf值计算平均值
            valid_r2 = result['R2'][~np.isinf(result['R2'])]
            valid_rmse = result['RMSE'][~np.isinf(result['RMSE'])]
            R2_out = valid_r2.mean() if len(valid_r2) > 0 else 0
            RMSE_out = valid_rmse.mean() if len(valid_rmse) > 0 else np.inf

        metrics_out = {
            'RMSE': RMSE_out,
            'R2': R2_out,
            'Accuracy': symbolic_accuracy
        }

        return metrics_out
