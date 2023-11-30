import os.path as osp
from typing import Dict, List, Optional

import mmengine
from datasets import Dataset
from mmengine.config import ConfigDict

from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.registry import ICL_PROMPT_TEMPLATES
from opencompass.utils import build_dataset_from_cfg, build_model_from_cfg
from opencompass.utils.logging import get_logger
from opencompass.utils.text_postprocessors import first_number_postprocess
from opencompass.utils.types import get_type_from_cfg


class LMEvaluator:
    """Evaluate output with language model.

    Args:
        prompt_template (ConfigDict): Prompt template configuration. Used to
            prompt the language model for scores. User can use two reserved
            keywords, ``{prediction}`` and ``{reference}``, referring to
            the prediction and optionally the reference answer.
        judge_cfg (ConfigDict): The config of language model as a judge.
        output_path (str): The path to prediction output.
        dataset_cfg (ConfigDict, optional): The config of the dataset to be
            evaluated.
        postprocessor (ConfigDict): The model prediction's postprocessor
            config.
    """

    def __init__(
        self,
        prompt_template: ConfigDict,
        judge_cfg: ConfigDict,
        output_path: str,
        cmp_order: Optional[str] = None,
        dataset_cfg: Optional[ConfigDict] = None,
        postprocessor: ConfigDict = dict(type=first_number_postprocess)
    ) -> None:
        self.output_path = output_path
        out_dir, out_name = osp.split(output_path)
        if not out_dir:
            out_dir = './'

        self.prompt_tmpl = ICL_PROMPT_TEMPLATES.build(prompt_template)

        max_out_len = judge_cfg.get('max_out_len', None)
        batch_size = judge_cfg.get('batch_size', None)
        model = build_model_from_cfg(model_cfg=judge_cfg)
        self.inferencer = GenInferencer(model,
                                        max_out_len=max_out_len,
                                        batch_size=batch_size,
                                        output_json_filepath=out_dir,
                                        output_json_filename=out_name)
        self.postprocessor = get_type_from_cfg(postprocessor)
        self.logger = get_logger()
        self.dataset_cfg = dataset_cfg
        assert cmp_order in [None, 'as-is', 'reversed', 'both']
        self.cmp_order = cmp_order

    def score(self, predictions, references: Optional[List] = None) -> Dict:
        if not isinstance(predictions[0], list):
            assert self.cmp_order is None, (
                'cmp_order must be None when '
                'only predictions from one model are '
                'provided.')
            predictions = [predictions]
        else:
            assert self.cmp_order, ('cmp_order must be specified when '
                                    'predictions from multiple models are '
                                    'provided.')
            if self.cmp_order == 'both':
                predictions = [
                    a + b for a, b in zip(predictions, reversed(predictions))
                ]
                if references:
                    references *= 2
            elif self.cmp_order == 'reversed':
                predictions.reverse()
                if references:
                    references.reverse()

        pred_dict = {}
        for i in range(len(predictions)):
            key = 'prediction' if i == 0 else f'prediction{i + 1}'
            pred_dict[key] = predictions[i]

        if self.dataset_cfg:
            dataset = build_dataset_from_cfg(self.dataset_cfg)
            if self.cmp_order == 'both':
                new_ds = {
                    k: dataset.test[k] * 2
                    for k in dataset.test.column_names
                }
                dataset.reader.dataset['test'] = Dataset.from_dict(new_ds)
            for k, v in pred_dict.items():
                dataset.reader.dataset['test'] = dataset.test.add_column(k, v)
                dataset.reader.input_columns.append(k)
            if references:
                dataset.reader.input_columns.append('reference')
                dataset.reader.dataset['test'] = dataset.test.add_column(
                    'reference', references)
        else:
            # build a default dataset just for comparison
            from opencompass.datasets.lmeval import LMEvalDataset
            input_columns = list(pred_dict.keys())
            if references:
                input_columns.append('reference')
            dataset = LMEvalDataset(reader_cfg=dict(
                input_columns=input_columns,
                output_column=None,
                train_split='test'),
                                    reference=references,
                                    **pred_dict)
        retriever = ZeroRetriever(dataset)
        self.inferencer.inference(retriever=retriever,
                                  prompt_template=self.prompt_tmpl)

        output = mmengine.load(self.output_path)
        return self.postprocess(output)

    def postprocess(self, output: Dict) -> Dict:
        """Postprocess output by adding necessary statistics or data into
        it."""
        if self.cmp_order is None:
            # Get average scores if the item is presented
            scores = []
            for k, v in output.items():
                score = self.postprocessor(v['prediction'])
                output[k]['score'] = score
                scores.append(score)
            try:
                output['score'] = sum(scores) / len(scores)
            except Exception:
                pass

        if self.cmp_order == 'both':
            half = len(output) // 2
            for k in list(output.keys())[:half]:
                output[k]['cmp_order'] = 'as-is'
            for k in list(output.keys())[half:]:
                output[k]['cmp_order'] = 'reversed'
        elif self.cmp_order in ['as-is', 'reversed']:
            for k in output.keys():
                output[k]['cmp_order'] = self.cmp_order

        return output
