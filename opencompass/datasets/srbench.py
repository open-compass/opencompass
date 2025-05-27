
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
import requests
import sympy as sp

@LOAD_DATASET.register_module()
class SRbenchDataset(BaseDataset):
    @staticmethod
    def load(path: str,local_mode=True):
        path="path_to_dataset"
        base_path = get_data_path(path,local_mode=local_mode) 
        formula_csv_path = os.path.join(base_path, f'FeynmanEquation_23.csv')
        data_files_base_dir = os.path.join(base_path, 'Feynman_with_units')
        processed_formulas_df = load_dataset('csv', data_files=formula_csv_path)['train']
        sample_data=[]
        prompt_1_out=[]
        prompt_2_out=[]
        for row in processed_formulas_df:
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
            if n_var == 2:
                prompt_1 = '\n'.join([f'x0={x1:.4f}, x1={x2:.4f}, y={y:.4f}' for x1, x2, y in sampled_data_i[:-1]])
                prompt_2=f'x0={sampled_data_i[-1, 0]:.4f}, x1={sampled_data_i[-1, 1]:.4f}, y={sampled_data_i[-1, 2]:.4f}'
            else:
                prompt_1 = '\n'.join([f'x0={x1:.4f}, x1={x2:.4f}, x2={x3:.4f},y={y:.4f}' for x1, x2,x3, y in sampled_data_i[:-1]])
                prompt_2=f'x0={sampled_data_i[-1, 0]:.4f}, x1={sampled_data_i[-1, 1]:.4f},x3={sampled_data_i[-1, 2]:.4f}, y={sampled_data_i[-1, 3]:.4f}'
            prompt_1_out.append(prompt_1)
            prompt_2_out.append(prompt_2)
        processed_formulas_df=processed_formulas_df.add_column(name="prompt1",column=prompt_1_out)
        processed_formulas_df=processed_formulas_df.add_column(name="prompt2",column=prompt_2_out)
        processed_formulas_df=processed_formulas_df.add_column(name="data_samples_list",column=sample_data)
        processed_formulas_df = processed_formulas_df.rename_column('n_variables', 'n_var')
        return processed_formulas_df

class SRbenchDatasetEvaluator(BaseEvaluator):
    def __init__(self,
            local_mode: bool = True,path=""):
            self.dataset=SRbenchDataset.load(path="",local_mode=local_mode)
    def _send_request(self,messages, mllm='4o'):      
        URL = f"your_api_url"
        API_KEY = "your_api_key"
        HEADERS = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {API_KEY}',
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
            'Content-Type': 'application/json'
        }
        model = mllm
        count = 0
        while True and count < 20:
            count += 1
            payload = json.dumps({
                "model": model,
                "messages": messages,
                "temperature": 0.6,
                "max_tokens": 50
            })
            session = requests.Session()
            session.keep_alive = False
            response = session.post(URL, headers=HEADERS, data=payload, verify=True)
            try:
                content = response.json()['choices'][0]['message']['content']
                break
            except:
                content=None
                pass

        return content
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
    def llm_evaluate(self,inferred_formula, true_formula, mllm='gpt-4o'):
        content = f'''
            You are given two mathematical formulas. Your task is to evaluate how structurally similar they are, and return a similarity score between 0 and 1.

            The score should reflect how closely the formulas match in terms of:
            - Mathematical operations and structure (e.g., same use of +, *, sin, etc.)
            - Term arrangement and complexity
            - Overall symbolic expression and intent

            A score of:
            - 1 means the formulas are structurally identical or mathematically equivalent
            - Around 0.8-0.9 means they are very similar but not identical
            - Around 0.5 means moderately similar (e.g., same overall shape but different terms)
            - Near 0 means structurally unrelated formulas

            Do not consider numerical evaluation or specific input values — only the symbolic structure and mathematical form.

            Formulas:
            Inferred Formula: {inferred_formula}
            True Formula: {true_formula}

            ONLY RETURN [THE SIMILARITY SCORE]
        '''
        messages = [{"role": "user", "content": content}]
        similarity_score = self._send_request(messages, mllm=mllm)
        #print(similarity_score)
        specific_emoji = "😊"
        if similarity_score.endswith(specific_emoji):
            similarity_score = similarity_score[:-len(specific_emoji)].rstrip()
        if similarity_score.startswith("["):
            similarity_score = similarity_score[1:]
        if similarity_score.endswith("]"):
            similarity_score = similarity_score[:-1]
        if similarity_score == ".":
            similarity_score= 0.0
        if similarity_score.endswith(specific_emoji):
            similarity_score = similarity_score[:-len(specific_emoji)].rstrip()
        return similarity_score
    
    def llm_translate(self,dirty_formula, mllm='gpt-4o'):
        content = f'''
            This is a language model's judgment on a mathematical formula. Please help me extract the mathematical formula from this judgment and return it:
            {dirty_formula}
            Please serve pi as pi and use x0, x1, x2,... to represent the variable names.
            ONLY RETURN THE FORMULA STRING (Not LATEX).
        '''
        messages = [{"role": "user", "content": content}]
        clean_formula = _send_request(messages, mllm=mllm)
        return clean_formula

    
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

        for row in range(len(references)):
            metrics['LLM_Score'] = float(self.llm_evaluate(predictions[row], references[row], mllm='gpt-4o'))
            n_var=self.dataset[row]["n_var"]
            y_true=references[row]
            func = self.parse_formula(predictions[row], n_var=n_var)
            if func is not None:
                try:
                    x_vars = [x[:, i] for i in range(n_var)]
                    y_pred = func(*x_vars)
                    if np.isscalar(y_pred):
                        y_pred = np.full_like(y_true, y_pred)
                    metrics['RMSE'] = root_mean_squared_error(y_true, y_pred)
                    metrics['R2'] = r2_score(y_true, y_pred)
                except Exception:
                    pass
            else:
                metrics["R2"]=0
                metrics["RMSE"]= root_mean_squared_error(y_true, y_pred)
            metrics['SymbolicMatch'] = self.is_symbolically_equivalent(predictions[row], references[row], n_var)
            result = result._append({
            'GT': references[row],
            'Pred': predictions[row],
            'Score': metrics['LLM_Score'],
            'RMSE': metrics['RMSE'],
            'R2': metrics['R2'],
            'SymbolicMatch': bool(metrics['SymbolicMatch'])
        }, ignore_index=True)

        if not result.empty:
            symbolic_accuracy = result['SymbolicMatch'].sum() / len(result)
            R2_out = result['R2'].sum() / len(result)
            Score_out = result['Score'].sum() / len(result)
            RMSE_out = result['RMSE'].sum() / len(result)
        metrics_out={
        'LLM_Score': Score_out,
        'RMSE': RMSE_out,
        'R2': R2_out,
        "Accuracy":symbolic_accuracy
        }
        return metrics_out
