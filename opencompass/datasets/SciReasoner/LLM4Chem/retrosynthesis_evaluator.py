# dataset: USPTO-50K
# https://github.com/otori-bird/retrosynthesis
# task : retrosynthesis prediction
import multiprocessing
import re
from functools import partial
from typing import Union

from rdkit import Chem, RDLogger
from tqdm import tqdm

from opencompass.openicl import BaseEvaluator
from opencompass.registry import TEXT_POSTPROCESSORS

# 关闭 RDKit 的冗余日志输出
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

# ----------------------------------------------------------------------
# 1. 复用原脚本的核心函数
# 我们将这些函数放在文件顶部，以便在 Evaluator 中调用
# ----------------------------------------------------------------------


def smi_tokenizer(smi):
    """
    Tokenizes a SMILES string using a regular expression.
    Note: This function was in the original script but is not directly used
    in the evaluation logic. It's included for completeness.
    """
    pattern = (r'(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)'
               r'|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])')
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens), f'SMILES tokenization failed for: {smi}'
    return ' '.join(tokens)


def canonicalize_smiles_clear_map(smiles, synthon=False, return_max_frag=True):
    """
    Canonicalizes a SMILES string, clears atom map numbers, and optionally
    returns the largest fragment.

    Args:
        smiles (str): The SMILES string to process.
        synthon (bool): Whether to skip the sanitization step.
        return_max_frag (bool): If True, returns a tuple of
        (full_smiles, max_frag_smiles).
        Otherwise, returns only the full SMILES.

    Returns:
        A tuple (str, str) or a single str depending on return_max_frag.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=not synthon)
    if mol is not None:
        # Clear atom map numbers
        for atom in mol.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                atom.ClearProp('molAtomMapNumber')
        try:
            smi = Chem.MolToSmiles(mol, isomericSmiles=True)
        except Exception:
            # Handle cases where MolToSmiles fails
            if return_max_frag:
                return '', ''
            else:
                return ''

        if return_max_frag:
            sub_smi_list = smi.split('.')
            if len(sub_smi_list) > 1:
                # Find the largest fragment
                sub_mols = [(s, Chem.MolFromSmiles(s, sanitize=not synthon))
                            for s in sub_smi_list]
                sub_mol_sizes = [(smi, len(m.GetAtoms()))
                                 for smi, m in sub_mols if m is not None]
                if sub_mol_sizes:
                    # Sort fragments by size and return the largest one
                    max_frag_smi = sorted(sub_mol_sizes,
                                          key=lambda x: x[1],
                                          reverse=True)[0][0]
                    # Recursively canonicalize the largest fragment
                    return smi, canonicalize_smiles_clear_map(
                        max_frag_smi, synthon=synthon, return_max_frag=False)
                else:
                    return smi, ''
            else:
                # If no fragments, the molecule is its own largest fragment
                return smi, smi
        else:
            return smi
    else:
        # If the molecule is invalid from the start
        if return_max_frag:
            return '', ''
        else:
            return ''


def compute_rank(prediction_group,
                 beam_size,
                 n_best,
                 score_alpha=1.0,
                 raw=False):
    """
    Ranks predictions for a single sample across multiple augmentations.

    Args:
        prediction_group (list): A 2D list of predictions for one sample,
                                 shaped [augmentation, beam_size].
                                 Each prediction is a tuple
                                 (full_smi, max_frag_smi).
        beam_size (int): The number of beams used in generation.
        n_best (int): The number of top predictions to consider.
        score_alpha (float): The scoring decay factor.
        raw (bool): If True, assumes no test augmentation (augmentation=1).

    Returns:
        A tuple containing:
        - A sorted list of ranked results: [(prediction_tuple, score), ...].
        - A list of invalid rates for each beam position.
    """
    rank = {}
    highest_pos = {}
    invalid_rates = [0] * beam_size

    if raw:
        # No test augmentation, len(prediction_group) is 1
        assert len(prediction_group) == 1, 'Raw mode requires augmentation=1'
        aug_predictions = prediction_group[0]
        for k in range(len(aug_predictions)):
            pred_tuple = aug_predictions[k]
            if not pred_tuple or not pred_tuple[0]:
                invalid_rates[k] += 1
                continue
            # Use rank as score for raw mode, lower is better
            rank[pred_tuple] = 1 / (score_alpha * k + 1)
    else:
        # With test augmentation
        for aug_predictions in prediction_group:
            valid_k = []  # Store valid (prediction_tuple, original_beam_index)
            for k, pred_tuple in enumerate(aug_predictions):
                if pred_tuple and pred_tuple[0]:
                    valid_k.append((pred_tuple, k))
                else:
                    invalid_rates[k] += 1

            # Deduplicate predictions within this augmentation run
            seen = set()
            deduped_preds = []
            for pred_tuple, k in valid_k:
                if pred_tuple not in seen:
                    seen.add(pred_tuple)
                    deduped_preds.append((pred_tuple, k))

            # Update ranks and highest positions
            for k, (pred_tuple, _) in enumerate(deduped_preds):
                score = 1 / (score_alpha * k + 1)
                rank[pred_tuple] = rank.get(pred_tuple, 0) + score
                highest_pos[pred_tuple] = min(
                    k, highest_pos.get(pred_tuple, float('inf')))

    # Combine scores for final ranking
    # The -1e8 term heavily penalizes lower ranks,
    # ensuring highest position is prioritized
    final_ranked_list = []
    if not raw:
        for key, score in rank.items():
            final_ranked_list.append((key, score + highest_pos[key] * -1e8))
    else:
        for key, score in rank.items():
            final_ranked_list.append((key, score))

    final_ranked_list.sort(key=lambda x: x[1], reverse=True)
    return final_ranked_list[:n_best], invalid_rates


# ----------------------------------------------------------------------
# 定义 Postprocessor (后处理器)
# ----------------------------------------------------------------------


@TEXT_POSTPROCESSORS.register_module()
def Retrosynthesis_postprocess(text: Union[str, None]) -> str:
    """
    从模型的原始输出中提取SMILES字符串。

    此函数会查找并返回被 <SMILES> 和 </SMILES> 标签包裹的内容。
    """
    # 检查输入是否为字符串，如果不是则返回空字符串，以提高代码健壮性
    if not isinstance(text, str):
        return ''

    # 删除 <think> </think> 标签及其内容
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # 使用正则表达式搜索SMILES标签内的内容
    # re.search() 会查找字符串中首次出现该模式的位置
    # (.*?) 是一个非贪婪捕获组，用于捕获两个标签之间的所有字符
    # re.DOTALL 标志让 '.' 可以匹配包括换行符在内的任意字符
    matches = re.findall(r'<SMILES>(.*?)</SMILES>', text, re.DOTALL)

    if matches:
        # 如果找到匹配项，group(1)会返回第一个捕获组的内容
        # .strip() 用于去除捕获内容前后可能存在的多余空格或换行
        return matches[-1].strip()
    else:
        # 如果没有找到匹配的模式，返回一个空字符串
        return ''


# ----------------------------------------------------------------------
# 定义 Evaluator (评估器) - 这是修改的核心
# ----------------------------------------------------------------------


class RetrosynthesisEvaluator(BaseEvaluator):
    """
    Evaluator for retrosynthesis models. It calculates Top-K accuracy and
    Max-Fragment accuracy based on SMILES string comparisons.
    """

    def __init__(self,
                 beam_size=10,
                 n_best=10,
                 augmentation=1,
                 score_alpha=1.0,
                 synthon=False,
                 process_number=None):
        super().__init__()
        self.beam_size = beam_size
        self.n_best = n_best
        self.augmentation = augmentation
        self.score_alpha = score_alpha
        self.synthon = synthon
        self.process_number = process_number if process_number is not None \
            else multiprocessing.cpu_count()
        print(f'Evaluator initialized with: beam_size={beam_size},'
              f' n_best={n_best}, augmentation={augmentation},'
              f' processes={self.process_number}')

    def score(self, predictions, references):
        """
        Calculates retrosynthesis prediction accuracy.

        Args:
            predictions (list): A flat list of predicted SMILES strings.
                                Shape: [data_size * augmentation * beam_size].
            references (list): A list of ground truth SMILES strings.
                               Shape: [data_size].

        Returns:
            dict: A dictionary containing evaluation metrics.
        """
        # flat predictions ->  1D
        print(f'len of predictions: {len(predictions)}')
        print(f'predictions[0]: {predictions[0]}')
        if isinstance(predictions, list):
            # Ensure predictions are a flat list
            if isinstance(predictions[0], list):
                predictions = [x for y in predictions for x in y]
            else:
                pass

        # print(f"predictions = {predictions} \nreferences = {references}")
        data_size = len(references)
        expected_preds_len = data_size * self.augmentation * self.beam_size
        if len(predictions) != expected_preds_len:
            return {
                'error':
                f'Length of predictions ({len(predictions)})'
                f' does not match expected length ({expected_preds_len})'
            }

        print('Canonicalizing predictions and references...')
        # Create a partial function for multiprocessing
        map_func = partial(canonicalize_smiles_clear_map,
                           synthon=self.synthon,
                           return_max_frag=True)

        with multiprocessing.Pool(self.process_number) as pool:
            can_predictions = list(
                tqdm(pool.imap(map_func, predictions),
                     total=len(predictions),
                     desc='Canonicalizing Predictions'))
            can_references = list(
                tqdm(pool.imap(map_func, references),
                     total=len(references),
                     desc='Canonicalizing References'))

        # Reshape the flat predictions list into a 3D list:
        # data_size x augmentation x beam_size
        predictions_reshaped = [[] for _ in range(data_size)]
        for i in range(data_size):
            for j in range(self.augmentation):
                start_idx = (i * self.augmentation + j) * self.beam_size
                end_idx = start_idx + self.beam_size
                predictions_reshaped[i].append(
                    can_predictions[start_idx:end_idx])

        # Initialize metric counters
        accuracy = [0] * self.n_best
        max_frag_accuracy = [0] * self.n_best
        total_invalid_rates = [0] * self.beam_size

        print('Computing ranks and accuracy...')
        is_raw_mode = (self.augmentation == 1)

        for i in tqdm(range(data_size), desc='Evaluating Samples'):
            prediction_group = predictions_reshaped[i]
            target_smi_tuple = can_references[i]

            # Skip evaluation for this sample if the ground truth is invalid
            if not target_smi_tuple or not target_smi_tuple[0]:
                continue

            ranked_results, invalid_rate = compute_rank(
                prediction_group,
                beam_size=self.beam_size,
                n_best=self.n_best,
                score_alpha=self.score_alpha,
                raw=is_raw_mode)

            # Aggregate invalid rates
            for j in range(len(invalid_rate)):
                total_invalid_rates[j] += invalid_rate[j]

            # Check for full molecule match
            found_match = False
            for j, (pred_tuple, _) in enumerate(ranked_results):
                if not found_match and pred_tuple[0] == target_smi_tuple[0]:
                    for k in range(j, self.n_best):
                        accuracy[k] += 1
                    found_match = True  # Ensure we only count the first match

            # Check for max fragment match
            found_frag_match = False
            for j, (pred_tuple, _) in enumerate(ranked_results):
                # Ensure max fragment is not empty before comparing
                if not found_frag_match and pred_tuple[1] and pred_tuple[
                        1] == target_smi_tuple[1]:
                    for k in range(j, self.n_best):
                        max_frag_accuracy[k] += 1
                    found_frag_match = True

        # Calculate final results
        results = {}
        # Usually, Top-1, 3, 5, 10 are reported
        for i in [k - 1 for k in [1, 3, 5, 10] if k <= self.n_best]:
            k = i + 1
            results[f'Top-{k} Accuracy'] = accuracy[i] / data_size * 100
            results[f'Top-{k} MaxFrag Accuracy'] = max_frag_accuracy[
                i] / data_size * 100

        # Report the invalid rate at the first beam position
        if self.beam_size > 0:
            total_predictions_at_beam1 = data_size * self.augmentation
            results['Invalid SMILES Rate (at beam 1)'] = (
                total_invalid_rates[0] / total_predictions_at_beam1 * 100) \
                if total_predictions_at_beam1 > 0 else 0

        return results


# Example Usage
if __name__ == '__main__':
    # --- Mock Data Generation ---
    # This simulates the kind of data the evaluator would receive.

    # Configuration
    BEAM_SIZE = 5
    N_BEST = 5
    AUGMENTATION = 3  # Use > 1 to test augmentation logic
    DATA_SIZE = 100

    # Ground truth molecules (references)
    mock_references = [
        'CCO.CN',  # Correct: CCO is largest fragment
        'c1ccccc1CC(=O)O',  # Correct
        'INVALID_SMILES',  # An invalid reference SMILES
        'CC(C)C(=O)N[C@@H](C)C(=O)O'  # Chiral molecule
    ] * (DATA_SIZE // 4)

    # Simulated model predictions (a flat list)
    mock_predictions = []
    for i in range(DATA_SIZE):
        target = mock_references[i]
        for _ in range(AUGMENTATION):
            # For each augmentation, create a beam of predictions
            beam = []
            # Make the first beam prediction correct for 20% of cases
            if i % 5 == 0:
                beam.append(target)
            else:
                beam.append('CC(C)=O')  # A common incorrect prediction

            # Add some other variations and invalid SMILES
            beam.append('c1cnccc1')
            beam.append('completely_invalid')  # Invalid
            # Add a prediction that only matches the largest fragment
            beam.append('CCO')
            # Fill the rest of the beam
            beam.extend(['C'] * (BEAM_SIZE - len(beam)))

            mock_predictions.extend(beam)

    print(f'Generated {len(mock_predictions)} '
          f'predictions for {len(mock_references)} references.')

    # --- Evaluation ---
    evaluator = RetrosynthesisEvaluator(
        beam_size=BEAM_SIZE,
        n_best=N_BEST,
        augmentation=AUGMENTATION,
        process_number=4  # Use 4 cores for the example
    )

    results = evaluator.score(mock_predictions, mock_references)

    # --- Print Results ---
    print('\n--- Evaluation Results ---')
    for key, value in results.items():
        print(f'{key}: {value:.2f}%')
    print('--------------------------\n')

    # --- Test RAW mode (no augmentation) ---
    print('Testing RAW mode (augmentation=1)...')
    evaluator_raw = RetrosynthesisEvaluator(
        beam_size=BEAM_SIZE,
        n_best=N_BEST,
        augmentation=1,  # RAW mode
        process_number=4)
    # Select only the first "augmentation" set of predictions
    mock_predictions_raw = []
    for i in range(DATA_SIZE):
        start_idx = i * AUGMENTATION * BEAM_SIZE
        end_idx = start_idx + BEAM_SIZE
        mock_predictions_raw.extend(mock_predictions[start_idx:end_idx])

    results_raw = evaluator_raw.score(mock_predictions_raw, mock_references)

    print('\n--- RAW Mode Evaluation Results ---')
    for key, value in results_raw.items():
        print(f'{key}: {value:.2f}%')
    print('---------------------------------\n')
