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
        base_path = get_data_path(path)
        formula_data_path = os.path.join(base_path, 'formula_data.json')
        dataset = load_dataset('json', data_files=formula_data_path)['train']

        sample_data = []
        prompt_1_out = []
        prompt_2_out = []
        for row in dataset:
            data = row['data']
            rand_idx = np.random.choice(len(data), 300, replace=False)
            points = np.array(data)[rand_idx]
            sample_data.append(points.tolist())
            length_data = points.shape[0]
            split_idx = int(length_data * 0.97)

            prompt_1 = change_data_to_prompt(points[:split_idx, :])
            prompt_2 = change_data_to_prompt(points[split_idx:, :])
            prompt_1_out.append(prompt_1)
            prompt_2_out.append(prompt_2)
        dataset = dataset.add_column(name='prompt1', column=prompt_1_out)
        dataset = dataset.add_column(name='prompt2', column=prompt_2_out)
        dataset = dataset.add_column(name='data_samples_list',
                                     column=sample_data)

        return dataset


def mydataset_postprocess(formula_str):
    # 1. 删除 Markdown 残留符号
    formula_str = formula_str.replace('×', '*').replace('·',
                                                        '*').replace('÷', '/')
    formula_str = formula_str.replace('−', '-').replace('^', '**')
    formula_str = formula_str.replace('"', '"').replace('"',
                                                        '"').replace('"', "'")

    # 2. 去除 markdown 反引号 ``` 和 $ 符号
    formula_str = formula_str.replace('`', '').replace('$', '').strip()

    # 3. 提取第一行公式（防止有多行解释性输出）
    formula_str = formula_str.split('\n')[0].strip()

    # 4. 用正则去除非合法字符（保留基本数学表达式）
    formula_str = re.sub(r'[^\w\s\+\-\*/\^\=\.\(\)]', '', formula_str)

    # 5. 确保左右去空格
    return formula_str.strip()


def change_data_to_prompt(points):
    data_prompt = ''
    for i in range(points.shape[0]):  # TODO 这行要根据变量数量改
        if points.shape[1] == 2:
            data_prompt += (f'x0={points[i, 0]:.5f}, '
                            f'y={points[i, 1]:.5f}\n')
        elif points.shape[1] == 3:
            data_prompt += (f'x0={points[i, 0]:.5f}, '
                            f'x1={points[i, 1]:.5f}, '
                            f'y={points[i, 2]:.5f}\n')
        elif points.shape[1] == 4:
            data_prompt += (f'x0={points[i, 0]:.5f}, '
                            f'x1={points[i, 1]:.5f}, '
                            f'x2={points[i, 2]:.5f}, '
                            f'y={points[i, 3]:.5f}\n')
        elif points.shape[1] == 5:
            data_prompt += (f'x0={points[i, 0]:.5f}, '
                            f'x1={points[i, 1]:.5f}, '
                            f'x2={points[i, 2]:.5f}, '
                            f'x3={points[i, 3]:.5f}, '
                            f'y={points[i, 4]:.5f}\n')
    return data_prompt


class SRbenchDatasetEvaluator(BaseEvaluator):

    def __init__(self, path=''):
        self.dataset = SRbenchDataset.load(path)

    def parse_formula(self, formula_str: str):
        try:
            if '=' in formula_str:
                expr_str = formula_str.split('=', 1)[1].strip()
            else:
                expr_str = formula_str.strip()

            if not expr_str:
                print(f"[Parse Error] 公式字符串为空或剥离后为空: '{formula_str}'")
                return None

            local_dict = {
                'sin': sp.sin,
                'cos': sp.cos,
                'exp': sp.exp,
                'sqrt': sp.sqrt,
                'log': sp.log,
                'arccos': sp.acos,
                'arcsin': sp.asin,
                'tan': sp.tan,
                'pi': sp.pi
            }
            expr = sp.sympify(expr_str, locals=local_dict)
            # 生成定义域
            variable_names = sorted([str(sym) for sym in expr.free_symbols])
            symbols = [sp.Symbol(name) for name in variable_names]
            for sym in symbols:
                local_dict[str(sym)] = sym
            # 转换为 numpy 表达式
            numpy_modules = [
                'numpy', {
                    'sqrt': np.sqrt,
                    'exp': np.exp,
                    'sin': np.sin,
                    'cos': np.cos,
                    'log': np.log,
                    'arcsin': np.arcsin,
                    'arccos': np.arccos,
                    'tan': np.tan,
                    'pi': np.pi
                }
            ]
            func = sp.lambdify(symbols, expr, modules=numpy_modules)
            return func, variable_names
        except (SyntaxError, TypeError, AttributeError, sp.SympifyError) as e:
            print(f'[Parse Error] 无法解析公式 "{formula_str}": {e}')
            return None
        except Exception as e:
            print(f'[Parse Error] 解析公式 "{formula_str}" 时发生意外错误: {e}')
            return None

    def generate_samples(self,
                         x0_range=(-10, 10),
                         x1_range=(-10, 10),
                         num_points=1000):
        """返回在定义域内的样本点 (x0, x1)"""
        x0_range = np.linspace(x0_range[0], x0_range[1], num_points)
        x1_range = np.linspace(x1_range[0], x1_range[1], num_points)
        x0, x1 = np.meshgrid(x0_range, x1_range)
        x0_vals = x0.flatten()
        x1_vals = x1.flatten()
        return x0_vals, x1_vals

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
            'RMSE': 100000.0,
            'NMSE': 100000.0,  # 新增：Normalized MSE
            'SymbolicMatch': False,
            'R2': -100000.0,
        }

        metrics_out = {
            'name': 'all',
            'mean_RMSE': 0,
            'mean_NMSE': 0,
            'mean_R2': 0,
            'SymbolicMatch': 0,
            'details': []
        }

        result = pd.DataFrame({
            'GT': pd.Series(dtype=str),
            'Pred': pd.Series(dtype=str),
            'RMSE': pd.Series(dtype=float),
            'NMSE': pd.Series(dtype=float),
            'R2': pd.Series(dtype=float),
            'SymbolicMatch': pd.Series(dtype=bool),
            'is_valid': pd.Series(dtype=bool)  # Add flag for valid predictions
        })

        # 结构评分（用 LLM）
        for row in range(len(references)):
            data = self.dataset[row]['data_samples_list']
            data = np.array(data)
            parse_result = self.parse_formula(predictions[row])

            # Initialize metrics for this prediction
            metrics['RMSE'] = 100000.0
            metrics['NMSE'] = 100000.0
            metrics['R2'] = -100000.0
            metrics['SymbolicMatch'] = False
            is_valid = False

            if parse_result is not None:
                func_pred, variable_names = parse_result
                func_gt, variable_names = self.parse_formula(references[row])
                var_num = len(variable_names)
                x, y_true = data[:, :var_num], data[:, -1]

                if func_pred is not None:
                    try:
                        x_vars = [x[:, i] for i in range(var_num)]
                        y_pred = func_pred(*x_vars)
                        if np.isscalar(y_pred):
                            y_pred = np.full_like(y_true, y_pred)

                        valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
                        y_true, y_pred = y_true[valid_mask], y_pred[valid_mask]

                        metrics['RMSE'] = root_mean_squared_error(
                            y_true, y_pred)
                        metrics['R2'] = r2_score(y_true, y_pred)
                        metrics['NMSE'] = np.mean(
                            (y_true - y_pred)**2) / np.var(y_true)
                        is_valid = True
                    except Exception as e:
                        print(f'Exception: {e}')
                        try:
                            x0_vals, x1_vals = self.generate_samples()
                            gt_vals = func_gt(x0_vals, x1_vals)
                            pred_vals = func_pred(x0_vals, x1_vals)
                            valid_mask = np.isfinite(gt_vals) & np.isfinite(
                                pred_vals)
                            gt_valid = gt_vals[valid_mask]
                            pred_valid = pred_vals[valid_mask]
                            metrics['RMSE'] = np.sqrt(
                                np.mean((gt_valid - pred_valid)**2))
                            # 计算 R2 值
                            metrics['R2'] = 1 - np.sum(
                                (gt_valid - pred_valid)**2) / np.var(gt_valid)
                            metrics['NMSE'] = np.mean(
                                (gt_valid - pred_valid)**2) / np.var(gt_valid)
                            is_valid = True
                        except Exception as e:
                            print(e)

                metrics['SymbolicMatch'] = self.is_symbolically_equivalent(
                    predictions[row], references[row], var_num)

            # Add to result DataFrame regardless of validity
            result = result._append(
                {
                    'GT': references[row],
                    'Pred': predictions[row],
                    'RMSE': metrics['RMSE'],
                    'NMSE': metrics['NMSE'],
                    'R2': metrics['R2'],
                    'SymbolicMatch': bool(metrics['SymbolicMatch']),
                    'is_valid': is_valid
                },
                ignore_index=True)

        # 添加每条数据的详细指标
        valid_count = 0
        for i in range(len(result)):
            metrics_out['details'].append({
                'index':
                i,
                'ground_truth':
                result.iloc[i]['GT'],
                'prediction':
                result.iloc[i]['Pred'],
                'RMSE':
                float(result.iloc[i]['RMSE']),
                'NMSE':
                float(result.iloc[i]['NMSE']),
                'R2':
                float(result.iloc[i]['R2']),
                'SymbolicMatch':
                bool(result.iloc[i]['SymbolicMatch']),
                'is_valid':
                result.iloc[i]['is_valid']
            })

            # Only count valid predictions in the final score
            if result.iloc[i]['is_valid']:
                metrics_out['mean_RMSE'] += result.iloc[i]['RMSE']
                metrics_out['mean_NMSE'] += result.iloc[i]['NMSE']
                metrics_out['mean_R2'] += result.iloc[i]['R2']
                metrics_out['SymbolicMatch'] += result.iloc[i]['SymbolicMatch']
                valid_count += 1

        # Calculate averages only for valid predictions
        if valid_count > 0:
            for key in metrics_out:
                if key != 'name' and key != 'details':
                    metrics_out[key] /= valid_count
        else:
            # If no valid predictions, set all metrics to default values
            metrics_out['mean_RMSE'] = 100000.0
            metrics_out['mean_NMSE'] = 100000.0
            metrics_out['mean_R2'] = -100000.0
            metrics_out['SymbolicMatch'] = 0

        return metrics_out
