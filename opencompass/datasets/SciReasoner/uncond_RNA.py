# flake8: noqa

import os
import re
import subprocess
from tempfile import TemporaryDirectory
from typing import Union

from datasets import Dataset

from opencompass.datasets.base import BaseDataset
from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS


@LOAD_DATASET.register_module()
class Uncond_RNA_Dataset(BaseDataset):

    @staticmethod
    def load(num, prompt):
        dataset = [{'input': prompt, 'output': ''} for _ in range(num)]
        return Dataset.from_list(dataset)


@TEXT_POSTPROCESSORS.register_module()
def RNA_postprocessor(text: Union[str, None]) -> str:
    if not text:
        return ''

    text = text.replace('T', 'U').replace('t', 'u')

    m = re.search(r'<rna>(.*?)</rna>', text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    m = re.search(r'^(.*?)</rna>', text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        before = m.group(1)
        # 从 before 中提取最后一段连续的 A/C/G/U（可按需求改成最长一段）
        chunks = re.findall(r'[ACGU]+', before, flags=re.IGNORECASE)
        return (chunks[-1].upper() if chunks else '').strip()

    return ''


class RNA_Evaluator(BaseEvaluator):

    def score(self, predictions, references):
        invalid_count = 0
        overlength_count = 0
        valid_rnas = []
        valid_bases = set('AUCG')
        avg_mfe = None
        rfam_families = []

        for idx, seq in enumerate(predictions):
            seq = seq.strip().upper()
            if not seq or any(base not in valid_bases for base in seq):
                invalid_count += 1
            else:
                valid_rnas.append((f'seq{idx}', seq))
                if len(seq) > 1024:
                    overlength_count += 1

        if len(predictions) == invalid_count:
            return {
                'total_samples': len(predictions),
                'invalid_prediction_count': invalid_count,
                'overlength_prediction_count': overlength_count,
                'valid_sequence_count': len(valid_rnas),
                'average_mfe': None,
                'retrieved_rfam_family_count': None,
            }

        with TemporaryDirectory() as tmpdir:
            fasta_path = os.path.join(tmpdir, 'valid_sequences.fasta')
            with open(fasta_path, 'w') as f:
                for seq_id, seq in valid_rnas:
                    f.write(f'>{seq_id}\n{seq}\n')

            mfe_file = self.run_rnafold(fasta_path, tmpdir)
            mfe_values = self.parse_mfe(mfe_file)
            avg_mfe = sum(mfe_values) / len(mfe_values) if mfe_values else None

            cache_dir = os.environ.get('COMPASS_DATA_CACHE', '')
            rfam_cm = os.path.join(cache_dir, 'Rfam/Rfam.cm')
            rfam_clanin = os.path.join(cache_dir, 'Rfam/Rfam.clanin')
            rfam_tblout = self.run_cmscan(fasta_path, tmpdir, rfam_cm,
                                          rfam_clanin)
            rfam_families = self.parse_unique_families(rfam_tblout)

        return {
            'total_samples': len(predictions),
            'invalid_prediction_count': invalid_count,
            'overlength_prediction_count': overlength_count,
            'valid_sequence_count': len(valid_rnas),
            'average_mfe': avg_mfe,
            'retrieved_rfam_family_count': len(rfam_families),
        }

    def run_rnafold(self, input_fasta, output_dir):
        output_file = os.path.join(output_dir, 'mfe_results.txt')
        cmd = (
            f'cd {output_dir} && RNAfold < '
            f'{os.path.abspath(input_fasta)} > {os.path.basename(output_file)} --noPS'
        )
        ret = subprocess.run(cmd, shell=True)
        if ret.returncode != 0:
            print(ret)
            raise RuntimeError('RNAfold execution failed!')
        return output_file

    def parse_mfe(self, output_file):
        mfe_values = []
        with open(output_file) as f:
            for line in f:
                match = re.search(r'\s\(([-\d\.]+)\)\s*$', line.strip())
                if match:
                    mfe = float(match.group(1))
                    mfe_values.append(mfe)
        return mfe_values

    def run_cmscan(self, fasta_file, output_dir, rfam_cm, rfam_clanin):
        tblout_path = os.path.join(output_dir, 'cmscan_results.tblout')
        cmscan_cmd = [
            'cmscan', '--rfam', '--cut_ga', '--nohmmonly', '--tblout',
            tblout_path, '--fmt', '2', '--clanin', rfam_clanin, rfam_cm,
            fasta_file
        ]
        result = subprocess.run(cmscan_cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise RuntimeError(f'cmscan failed:\n{result.stderr.decode()}')
        return tblout_path

    def parse_unique_families(self, tblout_file):
        families = set()
        with open(tblout_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) > 0:
                    family_id = parts[0]
                    families.add(family_id)
        return families
