# flake8: noqa: E501
# yapf: disable
import os.path as osp
import random
import re
from typing import Dict, List, Optional

import mmengine
from datasets import Dataset
from mmengine.config import ConfigDict

from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.registry import ICL_PROMPT_TEMPLATES
from opencompass.utils import build_dataset_from_cfg, build_model_from_cfg
from opencompass.utils.logging import get_logger


def extract_dicts(data):
    max_round_num = max(len(sublist) for sublist in data)
    predictions = [[] for _ in range(max_round_num)]
    for sublist in data:
        for i, d in enumerate(sublist):
            predictions[i].append(d.get('assistant'))
        for j in range(i + 1, max_round_num):
            predictions[j].append(None)
    return predictions


def order_preds_and_record_references(predictions, references, infer_order, seed=666):
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
        preds = [[pred['model_preds'][i], pred['model_name']] for pred in predictions]
        if infer_order == 'random':
            random.shuffle(preds)
        for j in range(len(preds)):
            list_of_preds[j].append(preds[j][0])
            references[i][f'answer{j+1}'] = preds[j][1]
    if infer_order == 'double':
        assert len(predictions) == 2
        list_of_preds = [a + b for a, b in zip(list_of_preds, reversed(list_of_preds))]
        reversed_references = []
        for item in references:
            reversed_item = item.copy()
            reversed_item['answer1'], reversed_item['answer2'] = reversed_item['answer2'], reversed_item['answer1']
            reversed_references.append(reversed_item)
        references += reversed_references
    return list_of_preds, references


def count_chinese_characters(text):
    words = re.findall(r'[\u4e00-\u9fff]', text)
    return len(words)


def count_english_words(text):
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    return len(words)


class LMEvaluator:
    """Evaluate output with language model.

    Args:
        prompt_template (ConfigDict): Prompt template configuration. Used to
            prompt the language model for scores. User can use two reserved
            keywords, ``{prediction}`` and ``{reference}``, referring to
            the prediction and optionally the reference answer.
        judge_cfg (ConfigDict): The config of language model as a judge.
        meta_review_prompt_template (ConfigDict, optional): Prompt template for meta judge model.
        output_path (str): The path to prediction output.
        dataset_cfg (ConfigDict, optional): The config of the dataset to be
            evaluated.
        pack_all_predictions (bool, optional): For multiround evaluation, judge all round or judge every single round.
        pred_postprocessor (ConfigDict): The model prediction's postprocessor
            config.
    """

    def __init__(
        self,
        prompt_template: ConfigDict,
        judge_cfg: ConfigDict,
        output_path: str,
        meta_review_prompt_template: Optional[ConfigDict] = None,
        pack_all_predictions: Optional[bool] = False,
        dataset_cfg: Optional[ConfigDict] = None,
        pred_postprocessor: Optional[ConfigDict] = None,
    ) -> None:
        self.output_path = output_path
        out_dir, out_name = osp.split(output_path)
        if not out_dir:
            out_dir = './'

        self.prompt_tmpl = ICL_PROMPT_TEMPLATES.build(prompt_template)
        if meta_review_prompt_template is not None:
            self.meta_review_prompt_tmpl = ICL_PROMPT_TEMPLATES.build(meta_review_prompt_template)

        max_out_len = judge_cfg.get('max_out_len', None)
        batch_size = judge_cfg.get('batch_size', None)
        model = build_model_from_cfg(model_cfg=judge_cfg)
        self.inferencer = GenInferencer(model,
                                        max_out_len=max_out_len,
                                        batch_size=batch_size,
                                        output_json_filepath=out_dir,
                                        output_json_filename=out_name)
        self.logger = get_logger()
        self.dataset_cfg = dataset_cfg
        self.pack_all_predictions = pack_all_predictions

    def score(self,
              predictions,
              judgements: Optional[List] = None,
              references: Optional[List] = None,
              meta: Optional[bool] = False,
              infer_order: Optional[str] = 'random') -> Dict:
        dup_indices = []
        if isinstance(predictions, list):
            """Apply to multi-model comparison."""
            if references is None:
                references = [{} for _ in range(len(predictions[0]['model_preds']))]
            predictions, references = order_preds_and_record_references(predictions, references, infer_order)

            # calculate dupicated predictions numbers
            total_predictions_num = len(predictions[0])

            # since there is impossible that two models response same pattern in multi-round chat, so we just check dup for single chat
            if isinstance(predictions[0][0], str):
                for i in range(len(predictions[0])):
                    check = [sub[i] for sub in predictions]
                    if len(set(check)) == 1:
                        dup_indices.append(i)

        elif isinstance(predictions, dict):
            """Apply to single-model scoring."""
            if references is None:
                references = [{} for _ in range(len(predictions[0]['model_preds']))]
            predictions = [predictions['model_preds']]

        # Due to the rarity of identical predictions, we have temporarily disabled the plagiarism detection feature.
        dup_indices = []

        if len(dup_indices) != 0:
            # remove dupicated predictions
            for index in sorted(dup_indices, reverse=True):
                for sublist in predictions:
                    del sublist[index]
                del references[index]

        pred_dict = {}
        if isinstance(predictions[0][0], str):
            # single chat for format like [['xxx', 'xxxx'], ['xxx', 'xxxx']]
            for i in range(len(predictions)):
                key = 'prediction' if i == 0 else f'prediction{i + 1}'
                gold_key = 'obj_gold'
                pred_dict[key] = predictions[i]
                pred_dict[gold_key] = references
                pred_dict[key + '_en_word_count'] = [count_english_words(j) for j in predictions[i]]
                pred_dict[key + '_cn_word_count'] = [count_chinese_characters(j) for j in predictions[i]]
            if judgements:
                for i in range(len(judgements)):
                    key = 'judgement' if i == 0 else f'judgement{i + 1}'
                    pred_dict[key] = judgements[i]['model_preds']
                    for j in range(len(references)):
                        references[j]['judge_model' + str(i + 1)] = judgements[i]['model_name']
        elif isinstance(predictions[0][0], list):
            # multi round for format like [[[{'round':1, 'user':'', 'assistant':''}, {'round':2, 'user':'', 'assistant':''}], [{'round':1, 'user':'', 'assistant':''}, {'round':2, 'user':'', 'assistant':''}]]]
            if self.pack_all_predictions:
                for i in range(len(predictions)):
                    key = 'prediction' if i == 0 else f'prediction{i + 1}'
                    pred_dict[key] = predictions[i]
            else:
                for i in range(len(predictions)):
                    multiround_predictions = extract_dicts(predictions[i])
                    for j in range(len(multiround_predictions)):
                        key = 'prediction' if i == 0 else f'prediction{i}'
                        key += '_r' + str(j + 1)
                        pred_dict[key] = multiround_predictions[j]
            if judgements:
                raise NotImplementedError(
                    'Not applied meta-reivew judge on multi-round dataset')
        else:
            raise NotImplementedError(f'{predictions[0][0]} with type {type(predictions[0][0])}, please check the postprocess you add to the prediction string is right or not, we suggest to return an empty string but not None')

        if self.dataset_cfg:
            dataset = build_dataset_from_cfg(self.dataset_cfg)

            if infer_order == 'double':
                new_ds = {k: dataset.test[k] * 2 for k in dataset.test.column_names}
                dataset.reader.dataset['test'] = Dataset.from_dict(new_ds)

            if len(dup_indices) != 0:
                remaining_indices = [idx for idx in range(len(dataset.test)) if idx not in dup_indices]
                dataset.reader.dataset['test'] = dataset.test.select(remaining_indices)
                print(f'Among total {total_predictions_num} predictions, there are {len(dup_indices)} predictions totally same, which are removed!')
            for k, v in pred_dict.items():
                dataset.reader.dataset['test'] = dataset.test.add_column(k, v)
                dataset.reader.input_columns.append(k)

            if references:
                dataset.reader.input_columns.append('reference')
                dataset.reader.dataset['test'] = dataset.test.add_column('reference', references)
        else:
            # build a default dataset just for comparison
            from opencompass.datasets.lmeval import LMEvalDataset
            input_columns = list(pred_dict.keys())
            if references:
                input_columns.append('reference')
            dataset = LMEvalDataset(
                reader_cfg=dict(input_columns=input_columns, output_column=None, train_split='test'),
                reference=references,
                **pred_dict
            )
        dataset.reader.output_column = 'reference'
        retriever = ZeroRetriever(dataset)

        if meta:
            self.inferencer.inference(retriever=retriever, prompt_template=self.meta_review_prompt_tmpl)
        else:
            self.inferencer.inference(retriever=retriever, prompt_template=self.prompt_tmpl)

        output = mmengine.load(self.output_path)
        return self.postprocess(output)

    def postprocess(self, output: Dict) -> Dict:
        """Postprocess output by adding necessary statistics or data into
        it."""
        return output
