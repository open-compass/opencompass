import os.path as osp
from typing import Dict, List, Optional
import random

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

def randomize_preds_and_record_references(predictions, references, random_order, seed=2680):
    random.seed(seed)
    list_of_preds = [[] for _ in range(len(predictions))]
    for i in range(len(predictions[0]['model_preds'])):
        preds = [[pred['model_preds'][i], pred['model_name']] for pred in predictions]
        if random_order:
            random.shuffle(preds)
        for j in range(len(preds)):
            list_of_preds[j].append(preds[j][0])
            references[i][f'answer{j+1}'] = preds[j][1]
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
        random_order: Optional[bool] = False,
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
        self.random_order = random_order

    def score(self, predictions, references: Optional[List] = None) -> Dict:
        if type(predictions) == list:
            '''Apply to multi-model comparison'''
            references = [{} for _ in range(len(predictions[0]['model_preds']))] if references is None else references
            predictions, references = randomize_preds_and_record_references(predictions, references, self.random_order)
        elif type(predictions) == dict:
            '''Apply to single-model scoring'''
            pass
        pred_dict = {}
        for i in range(len(predictions)):
            key = 'prediction' if i == 0 else f'prediction{i + 1}'
            pred_dict[key] = predictions[i]

        if self.dataset_cfg:
            dataset = build_dataset_from_cfg(self.dataset_cfg)
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
        dataset.reader.output_column='reference'
        #############TO DO: add remove same infer in dataset
        retriever = ZeroRetriever(dataset)
        self.inferencer.inference(retriever=retriever,
                                  prompt_template=self.prompt_tmpl)

        output = mmengine.load(self.output_path)
        return self.postprocess(output)



    def postprocess(self, output: Dict) -> Dict:
        """Postprocess output by adding necessary statistics or data into
        it."""
        return output