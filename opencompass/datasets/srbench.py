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
from torch import nn
import copy
import torch
from sympy import sympify, lambdify
from scipy.optimize import minimize
import torch.optim.adam
import math



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
    data_prompt = ""
    for i in range(points.shape[0]):  # TODO 这行要根据变量数量改
        if points.shape[1] == 2:
            data_prompt += f"""x0={points[i, 0]:.5f}, y={points[i, 1]:.5f}\n"""
        elif points.shape[1] == 3:
            data_prompt += f"""x0={points[i, 0]:.5f}, x1={points[i, 1]:.5f}, y={points[i, 2]:.5f}\n"""
        elif points.shape[1] == 4:
            data_prompt += f"""x0={points[i, 0]:.5f}, x1={points[i, 1]:.5f}, x2={points[i, 2]:.5f}, y={points[i, 3]:.5f}\n"""
        elif points.shape[1] == 5:
            data_prompt += f"""x0={points[i, 0]:.5f}, x1={points[i, 1]:.5f}, x2={points[i, 2]:.5f}, x3={points[i, 3]:.5f}, y={points[i, 4]:.5f}\n"""
    return data_prompt


class ConstantOptimizer(nn.Module):
    def __init__(self, expr, num_constants, X, y):
        super(ConstantOptimizer, self).__init__()
        self.num_constants = num_constants
        self.expr = expr
        self.x = X
        self.y = y

        # 1. 自动识别变量名：x0, x1, ..., xN
        var_names = sorted(set(re.findall(r"x\d+", expr)), key=lambda x: int(x[1:]))
        self.var_syms = [sp.Symbol(v) for v in var_names]
        self.var_names = var_names  # ['x0', 'x1', ...]
         # 逐个定义常数参数
        for i in range(num_constants):
            setattr(self, f'c{i}', nn.Parameter(torch.tensor(1.0, dtype=torch.float32, requires_grad=True), requires_grad=True))
        
        def to_tensor(x, dtype=torch.float32):
            return x if torch.is_tensor(x) else torch.tensor(x, dtype=dtype)

        # 3. 构造符号表达式和 lambdify 函数
        self.constant_syms = [sp.Symbol(f'c{i}') for i in range(num_constants)]
        expr = sp.sympify(sympify(expr))
        # expr = expr.subs("arcsin", "torch.arcsin")
        self.expr_fn = lambdify(self.constant_syms + self.var_syms, expr, modules=[{
            "sin": lambda x: torch.sin(to_tensor(x)),
            "cos": lambda x: torch.cos(to_tensor(x)),
            "exp": lambda x: torch.exp(to_tensor(x)),
            "log": lambda x: torch.log(to_tensor(x)),
            "sqrt": lambda x: torch.sqrt(to_tensor(x)),
            "tan": lambda x: torch.tan(to_tensor(x)),
            "Abs": lambda x: torch.abs(to_tensor(x)),
            "pow": lambda a, b: torch.pow(to_tensor(a), to_tensor(b)),
            "arcsin": lambda x: torch.asin(to_tensor(x)), 
            "arccos": lambda x: torch.acos(to_tensor(x)), 
        }])

    def forward(self):
        constants = [getattr(self, f'c{i}') for i in range(self.num_constants)]
        # constants = [c for c in self.constants]
        # inputs = []
        # for var in self.var_names:
        #     idx = int(var[1:])  # 从变量名 'x3' 得到索引 3
        #     inputs.append(self.x[:, idx])
        inputs = [self.x[:, i] for i in range(len(self.var_names))]
        y_pred = self.expr_fn(*constants, *inputs)
        # print(f"y_pred: {y_pred.shape}, y: {self.y.shape}")
        valid_mask = torch.isfinite(self.y) & torch.isfinite(y_pred)
        y_true = self.y[valid_mask]
        y_pred = y_pred[valid_mask]
        mean_y = torch.mean(y_true)
        denominator = torch.mean((y_true - mean_y) ** 2)
        numerator = torch.mean((y_pred - y_true) ** 2)
        loss = numerator / denominator
        return loss
def insert_constants(expr: str) -> str:
    var_pattern = re.compile(r'\bx\d+\b')  # 匹配变量 x0, x1 等
    const_counter = [0]  # 用于生成 c0, c1, ...

    def next_c():
        c_name = f"c{const_counter[0]}"
        const_counter[0] += 1
        return c_name

    # 第一步：为变量添加乘法系数（例如 x0 -> c0 * x0）
    def add_multiplicative_constants(match):
        return f"{next_c()} * {match.group()}"

    expr = var_pattern.sub(add_multiplicative_constants, expr)

    # 第二步：为部分括号内部表达式添加偏置项（例如 (c0 * x0 - c1 * x1) -> (c0 * x0 - c1 * x1 + c2)）
    # 处理 sin(...), log(...), 等函数内的表达式
    expr = re.sub(r'(\b(?:sin|cos|tan|log|exp|sqrt|arcsin|arccos|arctan)\s*\([^()]+\))',
                  lambda m: re.sub(r'\)(?![^\(]*\))', f' + {next_c()})', m.group(), count=1),
                  expr)
    expr = f"({expr}) + {next_c()}"
    return expr

def optimize_constants(expr, num_constants, X, y,learning_rate=1, epochs=10000, patience=5000):
    model = ConstantOptimizer(expr, num_constants, torch.tensor(X).cuda(), torch.tensor(y).cuda()).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float('inf')
    best_consts = torch.ones(num_constants, dtype=torch.float32)
    no_improve_epochs = 0
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()
        current_consts = [getattr(model, f'c{i}').detach().cpu().item() for i in range(num_constants)]
        if __name__ == '__main__':
            print(f"[Epoch {epoch}] Loss = {loss.item():.6f}")
        if loss.item() < best_loss - 1e-6:
            best_loss = loss.item()
            best_consts = current_consts
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                break
    return best_consts, best_loss

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


    def is_symbolically_equivalent(self, formula1, formula2, n_var=2):
        try:
            expr1 = sp.sympify(
                formula1.split('=')[1] if '=' in formula1 else formula1)
            expr2 = sp.sympify(
                formula2.split('=')[1] if '=' in formula2 else formula2)

            return sp.simplify(expr1 - expr2) == 0
        except Exception:
            return False

    def bfgs(self,pred_str, X, y, NMSE, n_restarts=10):
        idx_remove = True
        pred_str_raw = pred_str
        # pred_str = add_constant_to_formula(pred_str, model, tokenizer)
        pred_str = insert_constants(pred_str)
        print(f"添加常数后的公式: {pred_str}")
        # 拆分成变量（以 x 开头）和常数（以 c 开头）
        try:
            expr = sp.sympify(pred_str)
            # 获取所有符号
            all_symbols = expr.free_symbols
            const_symbols = sorted([s for s in all_symbols if str(s).startswith('c')], key=lambda x: int(str(x)[1:]))
            const_names = [str(s) for s in const_symbols]
            counstant_num = len(const_names)
        except Exception as e:
            candidate = re.sub(r"\bc\b", "constant", pred_str)
            expr = candidate
            for i in range(candidate.count("constant")):
                expr = expr.replace("constant", f"c{i}", 1)
            counstant_num = candidate.count("constant")
            pred_str = expr
        if idx_remove:
            bool_con = (X < 200).all(axis=1).squeeze()
            X = X[bool_con, :]
        best_overall_loss = float("inf")
        best_overall_const = None
        for i in range(n_restarts):
            print(f"\n[Restart {i + 1}/{n_restarts}]")
            consts, loss = optimize_constants(pred_str, counstant_num, X, y)
            if __name__ == '__main__':
                print(f"→ Restart {i + 1}: loss={loss:.6f}")
            if loss < best_overall_loss:
                best_overall_const = consts
                best_overall_loss = loss
            if best_overall_loss < 0.001:
                break
        print(best_overall_const, best_overall_loss)
        for i, val in enumerate(best_overall_const):
            if math.isnan(val):
                print(f"Warning: Constant {i} is NaN.")
                return pred_str_raw
        if best_overall_loss > NMSE:
            print("Warning: The optimized constants do not meet the NMSE requirement.")
            return pred_str_raw
        expr_new = pred_str
        for i, val in enumerate(best_overall_const):
            pattern = fr"\bc{i}\b"
            val_str = f"{val:.{32}f}"
            expr_new = re.sub(pattern, val_str, expr_new)
        return expr_new

    def score(self, predictions, references) -> dict:

        metrics = {
            'RMSE': 1,
            'NMSE': 1,  # 新增：Normalized MSE
            'SymbolicMatch': False,
            'R2': 0,
        }


        metrics_out = {
            'name': 'all',
            'mean_NMSE': 0,
            'mean_R2': 0,
            'details': []
        }


        result = pd.DataFrame({
            'GT': pd.Series(dtype=str),
            'Pred': pd.Series(dtype=str),
            'RMSE': pd.Series(dtype=float),
            'NMSE': pd.Series(dtype=float),
            'NMSE': pd.Series(dtype=float),
            'R2': pd.Series(dtype=float),
            'SymbolicMatch': pd.Series(dtype=bool),
            'is_valid': pd.Series(dtype=bool)  # Add flag for valid predictions
        })

        # 结构评分（用 LLM）
        # 结构评分（用 LLM）
        for row in range(len(references)):
            data = self.dataset[row]['data_samples_list']
            data = np.array(data)
            data_now= self.dataset[row]['data']
            data_now = np.array(data_now)

            formul_set=self.dataset[row]['set']
            gt_formula = references[row]
            infer_formula=predictions[row]
            parse_result = self.parse_formula(infer_formula)

            # Initialize metrics for this prediction
            metrics['RMSE'] = 1
            metrics['NMSE'] = 1
            metrics['R2'] = 0
            metrics['SymbolicMatch'] = False

            if parse_result is not None:
                try:
                    func_pred, variable_names = parse_result
                    func_gt, variable_names = self.parse_formula(references[row])

                    var_num = len(variable_names)
                    x, y_true = copy.deepcopy(data_now[:, :var_num]), copy.deepcopy(data_now[:, -1])
                    x_vars = [x[:, i] for i in range(var_num)]
                    y_pred = func_pred(*x_vars)
                    if np.isscalar(y_pred):
                        y_pred = np.full_like(y_true, y_pred)
                    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
                    y_true, y_pred = y_true[valid_mask], y_pred[valid_mask]
                    NMSE = np.mean((y_true - y_pred) ** 2) / (np.var(y_true) + 1e-8)

                    if NMSE > 0.001:
                        print(f"NMSE={NMSE}, 需要进行BFGS优化##############")
                        if '=' in infer_formula:
                            infer_formula = infer_formula.split('=', 1)[1].strip()
                        x, y_true = copy.deepcopy(data_now[:, :var_num]), copy.deepcopy(data_now[:, -1])
                        infer_formula = self.bfgs(infer_formula, x, y_true, NMSE)
                        
                        print(f"BFGS优化后的公式: {infer_formula}")
                    else:
                        print(f"NMSE={NMSE}, 不进行BFGS优化")
                except Exception as e:
                    print(f"False formula, may be constant input: {e}")
            
            if infer_formula is not None:
                try:
                    func_pred, variable_names = self.parse_formula(infer_formula)

                    func_gt, _ = self.parse_formula(gt_formula)
                    var_num = len(variable_names)
                    x, y_true = data[:, :var_num], data[:, -1]
                    x_vars = [x[:, i] for i in range(var_num)]
                    y_pred = func_pred(*x_vars)
                    if np.isscalar(y_pred):
                        y_pred = np.full_like(y_true, y_pred)

                    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
                    y_true, y_pred = y_true[valid_mask], y_pred[valid_mask]
                    RMSE = root_mean_squared_error(y_true, y_pred)
                    R2 = r2_score(y_true, y_pred)
                    NMSE = np.mean((y_true - y_pred) ** 2) / (np.var(y_true))
                    metrics['RMSE'] = RMSE if RMSE < metrics['RMSE'] else metrics['RMSE']
                    metrics['R2'] = R2 if R2 > metrics['R2'] else metrics['R2']
                    metrics['NMSE'] = NMSE if NMSE < metrics['NMSE'] else metrics['NMSE']
                    # 判断方程等价性
                    metrics['SymbolicMatch'] = self.is_symbolically_equivalent(infer_formula, gt_formula, var_num)
                except Exception as e:
                    print(f"Exception: {e}")

            # Add to result DataFrame regardless of validity
            result = result._append(
                {
                    'GT': gt_formula,
                    'set': formul_set,
                    'Pred': infer_formula,
                    'Pred': predictions[row],
                    'RMSE': metrics['RMSE'],
                    'NMSE': metrics['NMSE'],
                    'NMSE': metrics['NMSE'],
                    'R2': metrics['R2'],
                    'SymbolicMatch': bool(metrics['SymbolicMatch'])
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

            })

            # Only count valid predictions in the final score
            if result.iloc[i]['is_valid']:
                metrics_out['mean_NMSE'] += result.iloc[i]['NMSE']
                metrics_out['mean_R2'] += result.iloc[i]['R2']
                valid_count += 1

        # Calculate averages only for valid predictions
        if valid_count > 0:
            for key in metrics_out:
                if key != 'name' and key != 'details':
                    metrics_out[key] /= valid_count
        else:
            # If no valid predictions, set all metrics to default values
            metrics_out['mean_NMSE'] = 1
            metrics_out['mean_R2'] = 0

        return metrics_out
