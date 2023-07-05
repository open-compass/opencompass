from collections import defaultdict
from typing import Dict, List

import mmengine
from mmengine import ConfigDict, track_parallel_progress

from opencompass.registry import EVALUATORS, MODELS
from opencompass.utils import dataset_abbr_from_cfg, model_abbr_from_cfg


@EVALUATORS.register_module()
class ModelEvaluator:
    """TODO: Finish the implementation"""

    def __init__(
        self,
        config: ConfigDict,
    ) -> None:
        self.tasks = []
        self.cfg = config
        self.parse_cfg(self.cfg.pop('evaluator', ConfigDict({})))
        self.dataset_abbrs = [
            dataset_abbr_from_cfg(d) for d in self.cfg['datasets']
        ]
        self.model_abbrs = [model_abbr_from_cfg(m) for m in self.cfg['models']]
        assert len(self.model_abbrs) > 1

    def parse_cfg(self, cfg: ConfigDict):
        # The judger
        self.judger = MODELS.build(cfg['judger'])
        # Maximum number of workers
        self.max_num_workers = cfg.get('max_num_workers', 4)

    def evaluate(self):
        model_scores = defaultdict(int)
        all_partial_scores = track_parallel_progress(
            self._evaluate_dataset,
            self.dataset_abbrs,
            nproc=self.max_num_workers,
            keep_order=True)
        for partial_scores in all_partial_scores:
            for model_idx, score in partial_scores.items():
                model_scores[self.model_abbrs[model_idx]] += score
        print(model_scores)

    def _load_dataset(self, dataset_abbr: str):
        # for self.
        original_datasets = []
        self.responses: List[List[str]] = []
        self.questions: List[str] = []
        for model_abbr in self.model_abbrs:
            filename = f'output_model/{model_abbr}/{dataset_abbr}.json'
            original_datasets.append(mmengine.load(filename))
        for key in original_datasets[-1].keys():
            self.questions.append(original_datasets[-1][key]['origin_prompt'])
            responses = []
            for i in range(len(self.model_abbrs)):
                responses.append(original_datasets[i][key]['prediction'])
            self.responses.append(responses)

    def _evaluate_dataset(self, dataset_abbr: str):
        self._load_dataset(dataset_abbr=dataset_abbr)
        model_scores = defaultdict(int)
        for question, responses in zip(self.questions, self.responses):
            prompt = self._make_prompt(question, responses)
            print(prompt)
            output = self.judger.generate(prompt,
                                          max_out_len=2 *
                                          len(self.model_abbrs))
            model_scores = self._rank_models(output, model_scores)
        return model_scores

    def _make_prompt(self, question: str, responses: List[str]) -> str:
        prompt = ('Below are a question and a set of answers, each numbered by'
                  ' a digit. Please sort the answers from least to most '
                  'appropriate to the question. Only return the digit '
                  'seperated by a blank space. For example, when there are '
                  'three answers presented, you should say "1 0 2" when the '
                  'second answer is the best and the third is the worst.\n'
                  f'Q: {question}\n')
        for i, response in enumerate(responses):
            prompt += f'A{i + 1}: {response}\n'
        return prompt

    def _rank_models(self, output: str,
                     model_scores: defaultdict) -> Dict[str, int]:
        """Returns model ranking."""
        output = output.strip().split(' ')
        for score, model_idx in enumerate(output):
            model_scores[model_idx] += int(score)
        return model_scores
