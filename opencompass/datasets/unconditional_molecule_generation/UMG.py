import re

from datasets import Dataset, DatasetDict
from rdkit import Chem

from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class UMG_Dataset(BaseDataset):

    @staticmethod
    def load(train_path='', test_path='', max_cut=-1):
        gen_inst = 'Generate a molecule with <SMILES> '

        output_samples = [
            '<SMILES>CN1C=NC2=C1C(=O)N(C)C(=O)N2C</SMILES>',
            '<SMILES>c1ccccc1C(=O)O</SMILES>', '<SMILES>CCO</SMILES>',
            '<SMILES>CC(=O)Oc1ccccc1C(=O)O</SMILES>', '<SMILES>CCO</SMILES>'
        ]

        train_data = [{
            'input': gen_inst,
            'output': output,
        } for output in output_samples]

        len_test_data = 800

        if (max_cut != -1):
            len_test_data = min(len_test_data, max_cut)

        test_data = [{
            'input': gen_inst,
            'output': ''
        } for i in range(len_test_data)]

        dataset = DatasetDict({
            'train': Dataset.from_list(train_data),
            'test': Dataset.from_list(test_data)
        })
        return dataset


class UMG_Evaluator(BaseEvaluator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def is_valid_smiles_rdkit(self, s):
        """使用 RDKit 验证 SMILES 字符串"""
        if not isinstance(s, str) or not s:
            return False
        # 如果字符串中已经包含HTML标签样的结构，则认为它不是一个纯SMILES串
        # 这是为了避免重复处理已经被脚本标记过的SMILES
        if '<' in s or '>' in s:
            return False
        mol = Chem.MolFromSmiles(
            s, sanitize=False)  # sanitize=False 允许解析但可能化学上无效的SMILES
        return mol is not None

    def extract_smiles_simple(self, text: str) -> str | None:
        # match = re.search(r"⟨mol⟩([A-Za-z0-9()=#+@\\/\.-]+)⟨/mol⟩", text)
        if '<SMILES>' not in text:
            generic_pat = re.compile(r'(?<!\*)([A-Za-z0-9\u2080-\u2089'
                                     r'\(\)\.\+\-\=\#\$\:\@\*/%\\]{2,})(?!\*)')

            def generic_replace(m):
                candidate = m.group(1)

                if len(candidate) >= 4 and self.is_valid_smiles_rdkit(
                        candidate):
                    print('candidate', candidate)
                    return f'<SMILES> {candidate} </SMILES>'
                else:
                    return candidate

            text = generic_pat.sub(generic_replace, text)
        match = re.search(r'<SMILES> ([A-Za-z0-9()=#+@\\/\.-]+) </SMILES>',
                          text)
        if match:
            # 提取并打印出干净的结果
            clean_smiles = match.group(1)
            return clean_smiles
        else:
            return text

    def score(self, predictions):
        if not predictions:
            return {'validity': 0.0, 'uniqueness': 0.0, 'valid_smiles': []}
        valid_smiles = []
        for smiles in predictions:
            # RDKit有时会收到None或者空字符串，这里做一下防护
            if not smiles or not isinstance(smiles, str):
                continue
            smiles = self.extract_smiles_simple(smiles)
            # 核心步骤：使用RDKit检查SMILES是否有效
            mol = Chem.MolFromSmiles(smiles.strip())  # .strip()去除首尾空白
            if mol is not None:
                valid_smiles.append(smiles)

        total_generated = len(predictions)
        total_valid = len(valid_smiles)

        # 计算有效率 Validity = (有效SMILES数量 / 总生成SMILES数量)
        validity = float(total_valid) / float(
            total_generated) if total_generated > 0 else 0.0

        # 计算独特性 Uniqueness = (独特的有效SMILES数量 / 总有效SMILES数量)
        if total_valid > 0:
            unique_valid_smiles = set(valid_smiles)

            uniqueness = float(len(unique_valid_smiles)) / float(total_valid)
        else:
            uniqueness = 0.0
        print('validity', validity)
        print('uniquness', uniqueness)
        return {'validity': validity, 'uniquness': uniqueness}
