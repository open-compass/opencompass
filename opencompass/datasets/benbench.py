# flake8: noqa
# yapf: disable
import json
from typing import Dict, Optional

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class BenBenchDataset(BaseDataset):

    @staticmethod
    def load(path: str, tokenizer_path: str, tokenizer_kwargs: Optional[Dict] = dict(), num_gram: int=5, num_replica: int=5):
        import numpy as np
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, **tokenizer_kwargs)
        data = []
        with open(path, encoding='utf-8') as f:
            for index, line in enumerate(f):
                line = json.loads(line)
                if 'rewritten' in path:
                    text = line['rewritten_question'] + ' ' + line['rewritten_answer']
                elif 'origin' in path:
                    text = line['question'] + ' ' + line['answer']
                else:
                    raise ValueError(f'Unknown file type: {path}')
                tokens = tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) >= num_gram + max(num_replica, 2):
                    starting_points = np.linspace(2, len(tokens) - num_gram, num=num_replica, endpoint=True, dtype=int).tolist()
                else:
                    starting_points = np.linspace(2, max(2, len(tokens)), num=num_replica, endpoint=True, dtype=int).tolist()
                for s in starting_points:
                    data.append({
                        'index': index,
                        'prompt': tokenizer.decode(tokens[:s]),
                        'reference': tokenizer.decode(tokens[s:s+num_gram])
                    })
        dataset = Dataset.from_list(data)
        return dataset

def exact_match_score(predicted_text, original_text):
    return predicted_text == original_text

def edit_similarity_score(predicted_text, original_text):
    # Calculate normalized edit distance
    import editdistance

    edit_dist = editdistance.eval(predicted_text, original_text)
    max_length = max(len(predicted_text), len(original_text), 1)
    edit_similarity = 1 - (edit_dist / max_length)
    return edit_similarity

def rouge_l_score(predicted_text, original_text):
    # Calculate Rouge-L score
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_score = scorer.score(original_text, predicted_text)['rougeL'].fmeasure
    return rouge_score

@ICL_EVALUATORS.register_module()
class BenbenEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {'error': 'pred and refr length mismatch'}

        valid_exact_match, valid_edit_similarity, valid_rouge_score = 0, 0, 0
        total = len(predictions)
        for pred, ref in zip(predictions, references):
            exact_match = exact_match_score(pred, ref)
            edit_similarity = edit_similarity_score(pred, ref)
            rougeL = rouge_l_score(pred, ref)

            valid_exact_match += exact_match
            valid_edit_similarity += edit_similarity > 0.75
            valid_rouge_score += rougeL > 0.75

        return {
            'exact_match': valid_exact_match / total * 100,
            'edit_similarity': valid_edit_similarity / total * 100,
            'rougeL': valid_rouge_score / total * 100,
        }
