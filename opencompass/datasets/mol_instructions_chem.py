# flake8: noqa: W605
import json
import os
import re

from datasets import Dataset
from nltk.translate.meteor_score import meteor_score

from opencompass.openicl.icl_evaluator.icl_base_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path, get_logger

from .base import BaseDataset


@LOAD_DATASET.register_module()
class MolInstructionsDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        test_data = []
        path = os.path.join(get_data_path(path), name)
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                ins = json.loads(line)
                test_data.append({
                    'id': ins['id_ddm'],
                    'input': ins['dialogs'][0]['content'],
                    'output': ins['dialogs'][1]['content']
                })
        dataset = Dataset.from_list(test_data)
        return dataset


def extract_chem_tag(text, tag):
    pattern = re.compile(rf'<({tag})>(.*?)</\1>', re.DOTALL)
    matches = pattern.findall(text)
    if not matches:
        return None, None
    # 返回最后一个匹配的类型和内容
    last_match = matches[-1]
    return last_match[0], last_match[1].strip()  # (类型, 内容)


@ICL_EVALUATORS.register_module()
class FTSEvaluator(BaseEvaluator):
    """Exact match evaluator for name conversion."""

    def __init__(self, tag) -> None:
        super().__init__()
        self.tag = tag

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }

        valid_cnt = 0
        details = []
        for ori_pred, ori_ans in zip(predictions, references):
            pred = extract_chem_tag(ori_pred, self.tag)
            ans = extract_chem_tag(ori_ans, self.tag)
            if not pred[1]:
                pred = ('SMILES', ori_pred)
            if not ans[1]:
                ans = ('SMILES', ori_ans)
            detail = {'pred': pred[1], 'answer': ans[1]}
            # 将 SMILES 转换为 RDKit 分子对象
            if pred[0] == 'SELFIES':
                try:
                    import selfies as sf
                    pred = sf.decoder(pred[1])
                    ans = sf.decoder(ans[1])
                except:
                    detail['score'] = 0
                    details.append(detail)
                    continue
            else:
                pred = pred[1]
                ans = ans[1]
            from rdkit import Chem
            mol1 = Chem.MolFromSmiles(pred)
            mol2 = Chem.MolFromSmiles(ans)
            if mol1 is None or mol2 is None:
                detail['score'] = 0
                details.append(detail)
                continue
            valid_cnt += 1
            # 生成 Morgan 指纹（等同于 ECFP4）
            from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
            generator = GetMorganGenerator(radius=2, fpSize=2048)
            fp1 = generator.GetFingerprint(mol1)
            fp2 = generator.GetFingerprint(mol2)
            from rdkit.Chem import DataStructs
            similarity = DataStructs.TanimotoSimilarity(fp1, fp2) * 100
            detail['score'] = similarity
            details.append(detail)

        score = sum(detail['score'] for detail in details) / len(predictions)
        valid_score = valid_cnt / len(predictions) * 100

        return {'score': score, 'valid_score': valid_score, 'details': details}


def extract_number(text):
    pattern = re.compile(
        r'(?:<NUMBER>\s*|\\boxed\{)\s*(-?\d*\.?\d+)\s*(?:</NUMBER>|\})')
    matches = pattern.findall(text)
    if not matches:
        return None
    return [float(match) for match in matches][-1]


@ICL_EVALUATORS.register_module()
class MAEEvaluator(BaseEvaluator):
    """Exact match evaluator for name conversion."""

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }

        predictions = [
            extract_number(prediction) for prediction in predictions
        ]

        details = []
        for pred, ans in zip(predictions, references):
            if not pred:
                pred = 0.0
            detail = {'pred': pred, 'answer': float(ans)}
            mae_score = abs(pred - float(ans))
            detail['score'] = mae_score
            details.append(detail)

        score = sum(detail['score'] for detail in details) / len(predictions)

        return {'score': score, 'details': details}


@ICL_EVALUATORS.register_module()
class MeteorEvaluator(BaseEvaluator):
    """Exact match evaluator for name conversion."""

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        avg_score = 0
        details = []
        for pred, ans in zip(predictions, references):
            try:
                score = (meteor_score([ans.split()], pred.split())
                         if ans and pred else 0.0)
            except AttributeError:
                self.logger = get_logger()
                self.logger.warning(f'Failed to compute METEOR'
                                    f"score:\npred='{pred}'\nans='{ans}'")
                score = 0.0
            avg_score += score
            detail = {'pred': pred, 'answer': ans, 'score': score}
            details.append(detail)

        score = avg_score / len(predictions)

        return {'score': score, 'details': details}
