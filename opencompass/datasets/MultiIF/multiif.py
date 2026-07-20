import json

import pandas as pd
from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset
from .evaluation_main import (InputExample, test_instruction_following_loose,
                              test_instruction_following_strict)


@LOAD_DATASET.register_module()
class MultiIFDataset(BaseDataset):

    @staticmethod
    def load(path):
        path = get_data_path(path)
        df = pd.read_csv(path, keep_default_na=False)
        raw_data = []
        for _, row in df.iterrows():
            # Skip rows missing any turn (some samples only have 2 turns)
            if not row['turn_3_prompt']:
                continue
            dialogue = []
            reference = {'language': row['language']}
            for t in [1, 2, 3]:
                # user prompt
                prompt_msg = json.loads(row[f'turn_{t}_prompt'])
                dialogue.append(prompt_msg)
                # empty assistant = generation slot (filled by multiround)
                dialogue.append({'role': 'assistant', 'content': ''})
                # cumulative instruction list + kwargs
                reference[f'turn_{t}_instruction_id_list'] = json.loads(
                    row[f'turn_{t}_instruction_id_list'])
                # CSV kwargs is double-JSON: ["{}", "{\"k\":v}"]
                kwargs_raw = json.loads(row[f'turn_{t}_kwargs'])
                reference[f'turn_{t}_kwargs'] = [
                    json.loads(kw) if isinstance(kw, str) else kw
                    for kw in kwargs_raw
                ]
                # some instructions need the original prompt to
                # build_description
                reference[f'turn_{t}_prompt'] = prompt_msg['content']
            raw_data.append(dict(dialogue=dialogue, reference=reference))
        return Dataset.from_list(raw_data)


@ICL_EVALUATORS.register_module()
class MultiIFEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        """Score multi-turn instruction following.

        Args:
            predictions: List[List[str]] — 3 responses per sample
            references: List[dict] — instruction info per sample
        """
        results = {}
        turn_overalls = []

        for t in [1, 2, 3]:
            prompt_strict_correct, prompt_strict_total = 0, 0
            inst_strict_correct, inst_strict_total = 0, 0
            prompt_loose_correct, prompt_loose_total = 0, 0
            inst_loose_correct, inst_loose_total = 0, 0

            for pred_list, refer in zip(predictions, references):
                response = pred_list[t - 1]
                kwargs = [{k: v
                           for k, v in kw.items() if v is not None}
                          for kw in refer[f'turn_{t}_kwargs']]
                inp = InputExample(
                    key=0,
                    instruction_id_list=refer[f'turn_{t}_instruction_id_list'],
                    prompt=refer[f'turn_{t}_prompt'],
                    kwargs=kwargs,
                )

                # strict
                example = test_instruction_following_strict(inp, response)
                follow_list = example.follow_instruction_list
                prompt_strict_total += 1
                is_strict_correct = all(follow_list)
                prompt_strict_correct += is_strict_correct
                inst_strict_total += len(follow_list)
                inst_strict_correct += sum(follow_list)

                # loose
                example = test_instruction_following_loose(inp, response)
                follow_list = example.follow_instruction_list
                prompt_loose_total += 1
                is_loose_correct = all(follow_list)
                prompt_loose_correct += is_loose_correct
                inst_loose_total += len(follow_list)
                inst_loose_correct += sum(follow_list)

            ps = prompt_strict_correct / prompt_strict_total * 100
            isl = inst_strict_correct / inst_strict_total * 100
            pl = prompt_loose_correct / prompt_loose_total * 100
            ill = inst_loose_correct / inst_loose_total * 100
            turn_overall = (ps + isl + pl + ill) / 4

            results[f'turn_{t}_prompt_strict'] = ps
            results[f'turn_{t}_inst_strict'] = isl
            results[f'turn_{t}_prompt_loose'] = pl
            results[f'turn_{t}_inst_loose'] = ill
            results[f'turn_{t}_overall'] = turn_overall
            turn_overalls.append(turn_overall)

        results['overall'] = sum(turn_overalls) / len(turn_overalls)
        return results
