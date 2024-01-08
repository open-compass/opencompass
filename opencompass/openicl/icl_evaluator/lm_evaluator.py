# flake8: noqa: E501
import os.path as osp
import random
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


def order_preds_and_record_references(predictions,
                                      references,
                                      infer_order,
                                      seed=2680):
    """Order predictions based on args and recording regrading references.

    Args:
        predictions (List): List of multi model predictions.
        references (List): List of reference based on each problem.
        infer_order (str, optional): The mode of inference order.
        seed (int, optional): Random seed.
    """
    random.seed(seed)
    list_of_preds = [[] for _ in range(len(predictions))]
    for i in range(len(predictions[0]['model_preds'])):
        preds = [[pred['model_preds'][i], pred['model_name']]
                 for pred in predictions]
        if infer_order == 'random':
            random.shuffle(preds)
        for j in range(len(preds)):
            list_of_preds[j].append(preds[j][0])
            references[i][f'answer{j+1}'] = preds[j][1]
    if infer_order == 'double':
        assert len(predictions) == 2
        list_of_preds = [
            a + b for a, b in zip(list_of_preds, reversed(list_of_preds))
        ]
        reversed_references = []
        for item in references:
            reversed_item = item.copy()
            reversed_item['answer1'], reversed_item['answer2'] = reversed_item[
                'answer2'], reversed_item['answer1']
            reversed_references.append(reversed_item)
        references += reversed_references
    return list_of_preds, references


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
        infer_order: Optional[str] = 'random',
        dataset_cfg: Optional[ConfigDict] = None,
        postprocessor: ConfigDict = dict(type=first_number_postprocess)
    ) -> None:
        assert infer_order in ['random', 'double']
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
        self.infer_order = infer_order

    def score(self, predictions, references: Optional[List] = None) -> Dict:
        dup_indices = []

        if type(predictions) == list:
            """Apply to multi-model comparison."""
            references = [{} for _ in range(len(predictions[0]['model_preds']))
                          ] if references is None else references
            predictions, references = order_preds_and_record_references(
                predictions, references, self.infer_order)

            # calculate dupicated predictions numbers
            total_predictions_num = len(predictions[0])

            for i in range(len(predictions[0])):
                check = [sub[i] for sub in predictions]
                if len(set(check)) == 1:
                    dup_indices.append(i)

        elif type(predictions) == dict:
            """Apply to single-model scoring."""
            references = [{} for _ in range(len(predictions[0]['model_preds']))
                          ] if references is None else references
            predictions = [predictions['model_preds']]

        if len(dup_indices) != 0:
            # remove dupicated predictions
            for index in sorted(dup_indices, reverse=True):
                for sublist in predictions:
                    del sublist[index]
                del references[index]

        pred_dict = {}
        for i in range(len(predictions)):
            key = 'prediction' if i == 0 else f'prediction{i + 1}'
            pred_dict[key] = predictions[i]

        if self.dataset_cfg:
            dataset = build_dataset_from_cfg(self.dataset_cfg)

            if self.infer_order == 'double':
                new_ds = {
                    k: dataset.test[k] * 2
                    for k in dataset.test.column_names
                }
                dataset.reader.dataset['test'] = Dataset.from_dict(new_ds)

            if len(dup_indices) != 0:
                remaining_indices = [
                    idx for idx in range(len(dataset.test))
                    if idx not in dup_indices
                ]
                dataset.reader.dataset['test'] = dataset.test.select(
                    remaining_indices)
                print(
                    f'Among total {total_predictions_num} predictions, there are {len(dup_indices)} predictions totally same, which are removed!'
                )

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
        dataset.reader.output_column = 'reference'
        retriever = ZeroRetriever(dataset)
        self.inferencer.inference(retriever=retriever,
                                  prompt_template=self.prompt_tmpl)

        output = mmengine.load(self.output_path)
        return self.postprocess(output)

    def postprocess(self, output: Dict) -> Dict:
        """Postprocess output by adding necessary statistics or data into
        it."""
        return output
