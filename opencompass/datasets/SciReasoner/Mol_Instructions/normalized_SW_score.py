import math

def normalized_smith_waterman(seq1,
                              seq2,
                              matrix_name='BLOSUM45',
                              open_gap=-10,
                              extend_gap=-0.5):
    """
    Compute normalized Smith-Waterman score for protein sequences.

    Args:
        seq1, seq2 (str): Protein sequences (uppercase letters)
        matrix_name (str): Name of substitution matrix (default: BLOSUM62)
        open_gap (float): Gap opening penalty
        extend_gap (float): Gap extension penalty

    Returns:
        float: Normalized score between 0.0 and 1.0
    """

    from Bio.Align import PairwiseAligner, substitution_matrices

    # Initialize aligner
    aligner = PairwiseAligner()
    aligner.mode = 'local'  # Smith-Waterman algorithm
    aligner.open_gap_score = open_gap
    aligner.extend_gap_score = extend_gap

    # Load substitution matrix
    try:
        matrix = substitution_matrices.load(matrix_name)
    except ValueError:
        raise ValueError(f'Matrix {matrix_name} not available.'
                         f' Try: {substitution_matrices.load()}')

    # Set substitution matrix
    aligner.substitution_matrix = matrix

    # Calculate raw alignment score
    raw_score = aligner.score(seq1, seq2)
    if raw_score <= 0:
        return 0.0

    # Calculate self-alignment scores
    def calc_self_score(seq):
        """Calculate maximum possible self-alignment score"""
        score = 0
        for aa in seq:
            try:
                # Try direct lookup
                score += matrix[aa, aa]
            except KeyError:
                # Try reverse lookup for symmetric matrices
                score += matrix[aa, aa]  # Same residue
        return score

    self_score1 = calc_self_score(seq1)
    self_score2 = calc_self_score(seq2)

    # Handle invalid self-scores
    if self_score1 <= 0 or self_score2 <= 0:
        return 0.0

    # Compute normalization factor (geometric mean)
    norm_factor = math.sqrt(self_score1 * self_score2)

    return min(raw_score / norm_factor, 1.0)


# 示例用法
if __name__ == '__main__':
    # 示例序列（可以替换为实际的蛋白质序列）
    # target_sequence = "MGGKWSKSSIVGWPAVRERIRQTEPRTEPAA"  # 目标序列
    # generated_sequence = "MGGKWSKSSIVGWPAVRERIRRTEPAA"  # 模型生成的序列
    #
    # # target_sequence = 'MSTNPKPQRKTKRNTNRRPQDVKFPGGG'
    # # generated_sequence = 'MSTNPKPQRKTKRNTNRRPQDVK'
    #
    # # 计算归一化 SW 得分
    # normalized_score = calculate_normalized_sw_score(
    #     target_sequence,
    #     generated_sequence,
    #     gap_open=-10,
    #     gap_extend=-0.5,
    #     match_score=2,
    #     mismatch_score=-1
    # )
    #
    # print(f"归一化 SW 得分: {normalized_score:.3f}")
    #
    # # 计算归一化 Smith-Waterman 得分
    # normalized_sw_score = normalized_smith_waterman(
    #     target_sequence,
    #     generated_sequence,
    # )
    # print(f"归一化 Smith-Waterman 得分: {normalized_sw_score:.4f}")
    import json
    import os
    import re

    def Mol_Instructions_postprocess_Protein_Design(text, *args, **kwargs):
        """
        Extract the protein str between
        <protein> and </protein> in the sentences
        """
        text = text.strip()
        pattern = r'<protein>(.*?)</protein>'
        match = re.search(pattern, text)
        if match:
            text = match.group(1)
            # filter to make sure letters are all in the alphabet
            valid_letters = set('ACDEFGHIKLMNPQRSTVWY')
            text = ''.join(filter(lambda x: x in valid_letters, text))
        else:
            text = ''
        return text

    pred_list = []
    gt_list = []
    scores = []
    json_dir = (
        '/root/code/opencompass-sci/outputs/protein/mol_instructions/'
        '20250619_185027/predictions/qwen3-1.7B-sft-protein_0.7T_0.9p_50k')
    for filename in os.listdir(json_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(json_dir, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)
                for key, value in data.items():
                    pred = Mol_Instructions_postprocess_Protein_Design(
                        value['prediction'])
                    gt = Mol_Instructions_postprocess_Protein_Design(
                        value['gold'])
                    pred_list.append(pred)
                    gt_list.append(gt)
                    if not pred or not gt:
                        scores.append(0.0)
                    else:
                        # Calculate the normalized Smith-Waterman score
                        try:
                            score = normalized_smith_waterman(pred, gt)
                            scores.append(score)
                        except Exception:
                            import pdb

                            pdb.set_trace()
    import pdb

    pdb.set_trace()
