# molecule task
# https://github.com/zjunlp/Mol-Instructions/tree/main/evaluation/molecule

import json
import re

import numpy as np

from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download
from Levenshtein import distance as lev
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score

try:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem, MACCSkeys
except Exception:
    Chem, DataStructs, RDLogger, AllChem, MACCSkeys = None, None, None, None, None

try:
    import selfies as sf
except Exception:
    sf = None

from rouge_score import rouge_scorer
from sklearn.metrics import mean_absolute_error
from transformers import BertTokenizerFast

from opencompass.datasets.base import BaseDataset
from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.utils import get_data_path
import os

# RDLogger.DisableLog('rdApp.*')

@LOAD_DATASET.register_module()
class Mol_Instructions_Dataset(BaseDataset):

    @staticmethod
    def load(path, task, max_cut=-1, mini_set=False, hf_hub=False):

        # if (hf_hub is True):
        #     # load from huggingface hub
        #     train_data = []
        #     repo_id = test_path.split('/')[0] + '/' + test_path.split('/')[1]
        #     train_path = train_path.split(repo_id + '/')[1]
        #     test_path = test_path.split(repo_id + '/')[1]
        #
        #     train_path = hf_hub_download(repo_id,
        #                                  train_path,
        #                                  repo_type='dataset')
        #     test_path = hf_hub_download(repo_id,
        #                                 test_path,
        #                                 repo_type='dataset')

        path = get_data_path(path)
        train_path = os.path.join(path, f'{task}/dev/data.json')
        test_path = os.path.join(path, f'{task}/test/data.json')

        with open(train_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        with open(test_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        train_data = train_data[:5]
        # Limit the dataset to 5 samples for testing purposes

        if (max_cut != -1):
            test_data = test_data[:max_cut]
        if mini_set:
            import random
            random.seed(1024)
            test_data = random.sample(test_data, 150)
            random.seed()

        dataset = DatasetDict({
            'train': Dataset.from_list(train_data),
            'test': Dataset.from_list(test_data)
        })
        return dataset


def convert_to_canonical_smiles(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is not None:
        canonical_smiles = Chem.MolToSmiles(molecule,
                                            isomericSmiles=False,
                                            canonical=True)
        return canonical_smiles
    else:
        return None


@TEXT_POSTPROCESSORS.register_module()
def Mol_Instructions_postprocess_Mol(text, task, *args, **kwargs):
    """
    Filter end tokens in the sentences: "<|endoftext|>","<|im_end|>"
    """
    if task == 'property_prediction_str':
        # For property prediction, we only need the first line of the text
        text = text.strip()
        text = re.sub(r'<\|endoftext\|>', '', text)
        text = re.sub(r'<\|im_end\|>', '', text)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'.*?</think>\s*', '', text, flags=re.DOTALL)
        text = re.sub(r'(?<=\d) +(?=\d)|(?<=\.) +(?=\d)', '', text)
        num_match = re.search(r'[-+]?\d*\.\d+|\d+', text)
        text = num_match.group(0) if num_match else 0
    elif task in [
            'description_guided_molecule_design',
            'forward_reaction_prediction',
            'retrosynthesis',
            'reagent_prediction',
    ]:
        text = text.strip()
        text = re.sub(r'<\|endoftext\|>', '', text)
        text = re.sub(r'<\|im_end\|>', '', text)
        # first filter the <think></think> pattern

        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'.*?</think>\s*', '', text, flags=re.DOTALL)

        pattern = r'<SMILES>(.*?)</SMILES>'
        match = re.search(pattern, text)
        if match:
            smiles = match.group(1).strip()
            text = convert_to_canonical_smiles(smiles)
        else:
            # print('No SMILES found in the text. Using the original text.')
            # print(text)
            # import pdb; pdb.set_trace()
            text = None  # generate a false SMILES to avoid error in evaluation
    elif task in [
            'molecular_description_generation',
    ]:
        text = text.strip()
        text = re.sub(r'<\|endoftext\|>', '', text)
        text = re.sub(r'<\|im_end\|>', '', text)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'.*?</think>\s*', '', text, flags=re.DOTALL)

    return text


def compute_MAE_property_prediction_str(predictions, references):
    y_pred = np.array([float(p[0]) for p in predictions])
    y_true = np.array([float(r[0]) for r in references])
    mae = mean_absolute_error(
        y_true,
        y_pred) * 1000  # scale to match the presentation of Opencompass
    return {'mae': mae}


def compute_fingerprint_metricts(
    predictions,
    references,
    morgan_r=2,
):
    bad_mols = 0
    outputs = []

    for pred, refer in zip(predictions, references):
        try:
            if pred[0] is None:
                bad_mols += 1
                continue
            pred_ = Chem.MolFromSmiles(pred[0])
            refer_ = Chem.MolFromSmiles(refer[0])
            if pred_ is None:
                # print(pred)
                bad_mols += 1
                continue
            outputs.append((refer_, pred_))
        except Exception:
            import pdb
            pdb.set_trace()

    validity_score = len(outputs) / (len(outputs) + bad_mols)

    MACCS_sims = []
    morgan_sims = []
    RDK_sims = []

    enum_list = outputs

    for i, (gt_m, ot_m) in enumerate(enum_list):
        # if i % 100 == 0:
        #     if verbose: print(i, 'processed.')

        MACCS_sims.append(
            DataStructs.FingerprintSimilarity(
                MACCSkeys.GenMACCSKeys(gt_m),
                MACCSkeys.GenMACCSKeys(ot_m),
                metric=DataStructs.TanimotoSimilarity))
        RDK_sims.append(
            DataStructs.FingerprintSimilarity(
                Chem.RDKFingerprint(gt_m),
                Chem.RDKFingerprint(ot_m),
                metric=DataStructs.TanimotoSimilarity))
        morgan_sims.append(
            DataStructs.TanimotoSimilarity(
                AllChem.GetMorganFingerprint(gt_m, morgan_r),
                AllChem.GetMorganFingerprint(ot_m, morgan_r)))

    maccs_sims_score = np.mean(MACCS_sims)
    rdk_sims_score = np.mean(RDK_sims)
    morgan_sims_score = np.mean(morgan_sims)

    return {
        'validity_score': validity_score,
        'maccs_sims_score': maccs_sims_score,
        'rdk_sims_score': rdk_sims_score,
        'morgan_sims_score': morgan_sims_score
    }


def compute_mol_translation_selfies(predictions, references):
    outputs = []
    bad_mols = 0
    print(f'predictions: {predictions}, references: {references}')
    for pred, refer in zip(predictions, references):
        if pred[0] is None:
            bad_mols += 1
            continue
        pred_canonical_smiles = pred[0]
        refer_canonical_smiles = refer[0]
        try:
            pred_sf = sf.encoder(pred_canonical_smiles)
            refer_sf = sf.encoder(refer_canonical_smiles)
        except Exception:
            bad_mols += 1
            continue

        outputs.append(
            (refer_sf, pred_sf, refer_canonical_smiles, pred_canonical_smiles))

    references_self = []
    hypotheses_self = []

    references_smi = []
    hypotheses_smi = []

    for i, (gt_self, ot_self, gt_smi, ot_smi) in enumerate(outputs):
        gt_self_tokens = [c for c in gt_self]
        out_self_tokens = [c for c in ot_self]

        references_self.append([gt_self_tokens])
        hypotheses_self.append(out_self_tokens)

        gt_smi_tokens = [c for c in gt_smi]
        ot_smi_tokens = [c for c in ot_smi]

        references_smi.append([gt_smi_tokens])
        hypotheses_smi.append(ot_smi_tokens)

    # BLEU score
    if not references_self or not hypotheses_self:
        bleu_score_self = 0.0
    else:
        bleu_score_self = corpus_bleu(references_self, hypotheses_self)

    references_self = []
    hypotheses_self = []

    references_smi = []
    hypotheses_smi = []

    levs_self = []
    levs_smi = []

    num_exact = 0

    i = 0
    for i, (gt_self, ot_self, gt_smi, ot_smi) in enumerate(outputs):

        hypotheses_self.append(ot_self)
        references_self.append(gt_self)

        hypotheses_smi.append(ot_smi)
        references_smi.append(gt_smi)

        try:
            m_out = Chem.MolFromSmiles(ot_smi)
            m_gt = Chem.MolFromSmiles(gt_smi)

            if Chem.MolToInchi(m_out) == Chem.MolToInchi(m_gt):
                num_exact += 1
            # if gt == out: num_exact += 1
            # old version that didn't standardize strings
        except Exception:
            bad_mols += 1

        levs_self.append(lev(ot_self, gt_self))
        levs_smi.append(lev(ot_smi, gt_smi))

    # Exact matching score
    exact_match_score = num_exact / (i + 1)
    # if verbose:
    #     print('Exact Match:')
    #     print(exact_match_score)

    # Levenshtein score
    levenshtein_score_smi = np.mean(levs_smi)
    # if verbose:
    #     print('SMILES Levenshtein:')
    #     print(levenshtein_score_smi)

    return {
        'bleu_self_scores': bleu_score_self,
        'exact_match_score': exact_match_score,
        'levenshtein_score_smi': levenshtein_score_smi,
    }


def fix_smiles_brackets(smiles):
    """修复SMILES字符串中缺失的右括号"""
    if not isinstance(smiles, str):
        return smiles

    left_count = smiles.count('(')
    right_count = smiles.count(')')
    missing = left_count - right_count

    if missing > 0:
        return smiles + ')' * missing
    return smiles


class Mol_Instructions_Evaluator_Mol(BaseEvaluator):

    def __init__(self, task, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task = task

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        if not isinstance(predictions[0], list):
            predictions = [[pred] for pred in predictions]
        if not isinstance(references[0], list):
            references = [[ref] for ref in references]
        # import pdb;pdb.set_trace()
        task = self.task
        pred_list = predictions
        gold_list = references

        if task in ('property_prediction_str', ):
            results = compute_MAE_property_prediction_str(pred_list, gold_list)
        elif task in ('description_guided_molecule_design',
                      'forward_reaction_prediction', 'retrosynthesis',
                      'reagent_prediction'):
            fingerprint_metrics = compute_fingerprint_metricts(
                pred_list, gold_list)
            mol_translation_selfies = compute_mol_translation_selfies(
                pred_list, gold_list)
            # Combine the results from both computations
            results = {**fingerprint_metrics, **mol_translation_selfies}
            # change the order to
            # 'exact', 'blue', 'levenshtein', 'RDK',
            # 'MACCS', 'Morgan', 'validity'
            results = {
                'exact_match_score': results['exact_match_score'],
                'bleu_self_scores': results['bleu_self_scores'],
                'levenshtein_score_smi': results['levenshtein_score_smi'],
                'rdk_sims_score': results['rdk_sims_score'],
                'maccs_sims_score': results['maccs_sims_score'],
                'morgan_sims_score': results['morgan_sims_score'],
                'validity_score': results['validity_score']
            }
        elif task in ('molecular_description_generation', ):
            results = compute_text_translation_metrics(pred_list, gold_list)
        else:
            raise ValueError(task)

        return results


def compute_text_translation_metrics(
        predictions,
        references,
        text_model='allenai/scibert_scivocab_uncased',
        text_trunc_length=512):
    outputs = []

    for pred, refer in zip(predictions, references):
        try:
            pred_ = pred[0].rsplit('.', 1)[0] + '.' if isinstance(
                pred[0], str) else pred[0]
            outputs.append((refer[0], pred_))
        except Exception:
            import pdb
            pdb.set_trace()

    text_tokenizer = BertTokenizerFast.from_pretrained(text_model)

    meteor_scores = []

    references = []
    hypotheses = []

    for i, (gt, out) in enumerate(outputs):
        gt_tokens = text_tokenizer.tokenize(gt,
                                            truncation=True,
                                            max_length=text_trunc_length,
                                            padding='max_length')
        gt_tokens = list(filter(('[PAD]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[CLS]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[SEP]').__ne__, gt_tokens))

        out_tokens = text_tokenizer.tokenize(out,
                                             truncation=True,
                                             max_length=text_trunc_length,
                                             padding='max_length')
        out_tokens = list(filter(('[PAD]').__ne__, out_tokens))
        out_tokens = list(filter(('[CLS]').__ne__, out_tokens))
        out_tokens = list(filter(('[SEP]').__ne__, out_tokens))

        references.append([gt_tokens])
        hypotheses.append(out_tokens)

        mscore = meteor_score([gt_tokens], out_tokens)
        meteor_scores.append(mscore)

    bleu2 = corpus_bleu(references, hypotheses, weights=(.5, .5))
    bleu4 = corpus_bleu(references, hypotheses, weights=(.25, .25, .25, .25))

    _meteor_score = np.mean(meteor_scores)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    rouge_scores = []

    references = []
    hypotheses = []

    for i, (gt, out) in enumerate(outputs):
        rs = scorer.score(out, gt)
        rouge_scores.append(rs)

    rouge_1 = np.mean([rs['rouge1'].fmeasure for rs in rouge_scores])
    rouge_2 = np.mean([rs['rouge2'].fmeasure for rs in rouge_scores])
    rouge_l = np.mean([rs['rougeL'].fmeasure for rs in rouge_scores])

    return {
        'bleu2': bleu2,
        'bleu4': bleu4,
        'meteor_score': _meteor_score,
        'rouge1': rouge_1,
        'rouge2': rouge_2,
        'rougeL': rouge_l
    }
