import re

from datasets import Dataset, DatasetDict

from opencompass.datasets.base import BaseDataset
from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS


@LOAD_DATASET.register_module()
class UPGDataset(BaseDataset):

    @staticmethod
    def load(tag_bool=True, max_cut=-1):
        if tag_bool:
            gen_inst = 'Generate a protein sequence with <protein> </protein>.'
        else:
            gen_inst = 'Generate a protein sequence.'
        output_samples = [
            '<protein>MGDVEKGKKIFIMKCSQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGYSYTAANKNK'
            'GIIWGEDTLMEYLENPKKYIPGTKMIFVGIKKKEERADLIAYLKKATNE</protein>',
            '<protein>MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKL'
            'PVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFE'
            'GDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIED'
            'GSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAG'
            'ITLGMDELYK</protein>',
            '<protein>MKALIVLGLVLLSVTVQGKVFERCELARTLKRLGMDGYRGISLANWMCLAKWESGY'
            'NTRATNYNAGDRSTDYGIFQINSRYWCNDGKTPGAVNACHLSCSALLQDNIADAVACAKRVVRD'
            'PQGIRAWVAWRNRCQNRDVRQYVQGCGV</protein>',
            '<protein>MLEVKERIAQAKAEIPAPVELAPEEIERLLWKLGWRPVAYGSEEKARELDELYGHP'
            'FAQEHPKEGAAGPVLAAARGGLEEYGAVEWGWGLGEREWAAAGRVAADVVRRLDGEAREGTLPA'
            'EAEAFPALAAALEHHHHHH</protein>',
            '<protein>MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRR'
            'EAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN</protein>',
        ]

        train_data = [{
            'input': gen_inst,
            'output': output,
        } for output in output_samples]

        len_test_data = 1000
        # len_test_data = 10

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


@TEXT_POSTPROCESSORS.register_module('UPG_postprocess')
def UPG_postprocess(text):
    # Check if the input is a string;
    # if not, return an empty string to improve robustness
    if not isinstance(text, str):
        return ''

    # re.findall() searches for all occurrences of the pattern in the string
    # (.*?) is a non-greedy capture group,
    # capturing everything between the two tags
    # re.DOTALL flag makes '.' match any character, including newlines
    matches = re.findall(r'<protein>(.*?)</protein>', text, re.DOTALL)

    if matches:
        # If a match is found, take the last one
        # and strip leading/trailing whitespace
        s = matches[-1].strip()
        # Remove ';'
        s = s.replace(';', '')
        # Remove spaces
        s = s.replace(' ', '')

        def clean_prediction(seq: str) -> str:
            valid = set('ACDEFGHIKLMNPQRSTVWY-'
                        )  # Valid uppercase amino acid characters
            return ''.join([aa for aa in seq.upper() if aa in valid])

        s = clean_prediction(s)
        return s
    else:
        # If no match is found, return an empty string
        return ''


class UPG_Evaluator(BaseEvaluator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _calculate_sequence_identity(self, seq1, seq2):
        """
        Calculate sequence identity between two sequences.
        This is a simplified implementation for sequences of equal length,
        computed by direct position-wise comparison.
        More accurate methods may require alignment algorithms
        (e.g., Smith-Waterman).
        """
        if len(seq1) != len(seq2) or not seq1:
            # For unequal-length or empty sequences, treat identity as 0
            # or adopt a more complex alignment strategy if needed.
            # Here we return 0 for simplicity.
            return 0
        matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
        return matches / len(seq1)

    def score(self, predictions, references=None):
        """
        Evaluate the generated protein sequences.

        Args:
            predictions (list[str]): List of model-generated protein sequences.
            references (list[str], optional):
            Reference sequences; ignored here.

        Returns:
            dict: Dictionary containing evaluation metrics.
        """
        if not predictions:
            return {
                'num_length_less_100': 0,
                'valid_rate': 0,
                'average_length': 0,
                'diversity': 0,
                'average_plddt': 0,
                'info': 'Input predictions list is empty.'
            }

        ori_len = len(predictions)

        predictions = [pred for pred in predictions if len(pred) > 0]
        predictions = [
            pred for pred in predictions if not (pred.strip() == '')
        ]

        if len(predictions) == 0:
            return {
                'num_length_less_100': 0,
                'valid_rate': 0,
                'average_length': 0,
                'diversity': 0,
                'average_plddt': 0,
                'info': 'Input predictions list is empty.'
            }

        valid_rate = len(predictions) / ori_len

        # --- 1. Compute Average Length ---
        total_length = sum(len(seq) for seq in predictions)
        avg_length = total_length / len(predictions)

        # --- 2. Compute Diversity ---
        # Use a greedy clustering algorithm with 99%
        # sequence identity threshold
        clusters_representatives = []
        for seq in predictions:
            is_in_existing_cluster = False
            for representative in clusters_representatives:
                # Note: This uses simplified equal-length identity calculation.
                # For sequences of different lengths,
                # use sequence alignment tools.
                # As a simple strategy, compare only if lengths are close.
                if abs(
                        len(seq) - len(representative)
                ) < 20:  # Only compare sequences with small length differences
                    if self._calculate_sequence_identity(
                            seq,
                            representative) >= 0.99:  # 99% sequence identity
                        is_in_existing_cluster = True
                        break
            if not is_in_existing_cluster:
                clusters_representatives.append(seq)

        num_clusters = len(clusters_representatives)
        diversity = num_clusters / len(predictions)

        # --- 3. Compute Average pLDDT ---
        # Only compute for sequences shorter than 100 residues
        plddt_scores = []
        sequences_for_plddt = [
            seq for seq in predictions if (len(seq) < 100 and len(seq) > 0)
        ]

        for s in sequences_for_plddt:
            print(s)

        if sequences_for_plddt:
            from .omegafold.__main__ import main as plddt_main
            plddt_scores = plddt_main(sequences_for_plddt)
            avg_plddt = sum(plddt_scores) / len(plddt_scores)
        else:
            avg_plddt = 0.0  # If no sequences shorter than 100, set to 0

        return {
            'num_length_less_100': len(sequences_for_plddt),
            'valid_rate': round(valid_rate, 4),
            'average_length': round(avg_length, 2),
            'diversity': round(diversity, 4),
            'average_plddt': round(avg_plddt, 2)
        }
