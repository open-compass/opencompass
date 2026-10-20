# flake8: noqa: E501
import os.path as osp
import random
import re
from typing import Dict, List, Optional, Union

import mmengine
from datasets import Dataset
from mmengine.config import ConfigDict

from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.registry import DICT_POSTPROCESSORS, ICL_PROMPT_TEMPLATES
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


def order_preds_and_record_references(
    predictions: List,
    references: List,
    infer_order: List,
    seed: int = 666,
    keep_preds: bool = False,
    base_model_abbrs: List[str] = None,
):
    """Order predictions based on args and recording regrading references.

    Args:
        predictions (List): List of multi model predictions.
        references (List): List of reference based on each problem.
        infer_order (str, optional): The mode of inference order.
        seed (int, optional): Random seed.
        keep_preds (bool, optional): Whether to save model predictions in references. This will be available as input in postprocessor. Defaults to False.
        base_model_abbrs (List[str], optional): List of base models passed from dataset cfg.
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

            if keep_preds:
                references[i][f'prediction{j+1}'] = preds[j][0]

        if base_model_abbrs is not None:
            if isinstance(base_model_abbrs, str):
                base_model_abbrs = [base_model_abbrs]

            references[i]['base_models'] = base_model_abbrs

    if infer_order == 'double':
        assert len(predictions) == 2
        list_of_preds = [
            a + b for a, b in zip(list_of_preds, reversed(list_of_preds))
        ]
        reversed_references = []
        for item in references:
            reversed_item = item.copy()
            reversed_item['answer1'], reversed_item['answer2'] = (
                reversed_item['answer2'],
                reversed_item['answer1'],
            )

            if keep_preds:
                reversed_item['prediction1'], reversed_item['prediction2'] = (
                    reversed_item['prediction2'],
                    reversed_item['prediction1'],
                )

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
        keep_predictions (bool): Whether to save model predictions in references. Useful when postprocessor requires model predictions as input to calculate additional features (e.g. response length, markdown list counts, ...). Defaults to False.
        multi_eval (bool): Whether to do multiple evaluation with different prompt settings.
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
        dict_postprocessor: Optional[ConfigDict] = None,
        keep_predictions: bool = False,
        multi_eval: bool = False,
    ) -> None:
        self.multi_eval = multi_eval
        self.output_path = output_path
        out_dir, out_name = osp.split(output_path)
        if not out_dir:
            out_dir = './'

        self.prompt_tmpl = ICL_PROMPT_TEMPLATES.build(prompt_template)
        if meta_review_prompt_template is not None:
            self.meta_review_prompt_tmpl = ICL_PROMPT_TEMPLATES.build(
                meta_review_prompt_template)

        max_out_len = judge_cfg.get('max_out_len', None)
        batch_size = judge_cfg.get('batch_size', None)
        model = build_model_from_cfg(model_cfg=judge_cfg)
        self.inferencer = GenInferencer(
            model,
            max_out_len=max_out_len,
            batch_size=batch_size,
            output_json_filepath=out_dir,
            output_json_filename=out_name,
        )
        self.logger = get_logger()
        self.dataset_cfg = dataset_cfg
        self.pack_all_predictions = pack_all_predictions
        self.pred_postprocessor = pred_postprocessor
        self.dict_postprocessor = dict_postprocessor
        self.keep_predictions = keep_predictions

    def score(
        self,
        predictions,
        judgements: Optional[List] = None,
        references: Optional[List] = None,
        meta: Optional[bool] = False,
        infer_order: Optional[str] = 'random',
    ) -> Dict:
        dup_indices = []
        if isinstance(predictions, list):
            """Apply to multi-model comparison."""
            if references is None:
                references = [
                    {} for _ in range(len(predictions[0]['model_preds']))
                ]

            base_model_abbrs = None
            if self.dataset_cfg is not None:
                if 'base_models' in self.dataset_cfg:
                    base_models = self.dataset_cfg['base_models']

                    if isinstance(base_models, Dict):
                        base_models = [base_models]

                    base_model_abbrs = [
                        base_mdl['abbr'] for base_mdl in base_models
                    ]

            predictions, references = order_preds_and_record_references(
                predictions=predictions,
                references=references,
                infer_order=infer_order,
                keep_preds=self.keep_predictions,
                base_model_abbrs=base_model_abbrs,
            )

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
                references = [
                    {} for _ in range(len(predictions[0]['model_preds']))
                ]
            if self.multi_eval:
                assert references is not None
                assert 'judge_prompt_list' in references[0]
                self.multi_eval_times = len(references[0]['judge_prompt_list'])
                temp_predictions_save_list = []
                for idx, pred in enumerate(predictions['model_preds']):
                    for judge_prompt in references[idx]['judge_prompt_list']:
                        temp_prediction = judge_prompt.replace(
                            '{prediction}', pred)
                        temp_predictions_save_list.append(temp_prediction)
                predictions['model_preds'] = temp_predictions_save_list

                temp_references_save_list = []
                for item in references:
                    new_item = {
                        key: value
                        for key, value in item.items()
                        if key != 'judge_prompt_list'
                    }
                    if 'judge_prompt_list' in item:
                        for prompt in item['judge_prompt_list']:
                            temp_item = new_item.copy()
                            temp_item['judge_prompt'] = prompt
                            temp_references_save_list.append(temp_item)
                    else:
                        temp_references_save_list.append(item)
                references = temp_references_save_list
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
                pred_dict[key + '_en_word_count'] = [
                    count_english_words(j) for j in predictions[i]
                ]
                pred_dict[key + '_cn_word_count'] = [
                    count_chinese_characters(j) for j in predictions[i]
                ]
            if judgements:
                for i in range(len(judgements)):
                    key = 'judgement' if i == 0 else f'judgement{i + 1}'
                    pred_dict[key] = judgements[i]['model_preds']
                    for j in range(len(references)):
                        references[j]['judge_model' +
                                      str(i + 1)] = judgements[i]['model_name']
        elif isinstance(predictions[0][0], list):
            # multi round for format like [[[{'round':1, 'user':'', 'assistant':''}, {'round':2, 'user':'', 'assistant':''}], [{'round':1, 'user':'', 'assistant':''}, {'round':2, 'user':'', 'assistant':''}]]]
            if self.pack_all_predictions:
                for i in range(len(predictions)):
                    key = 'prediction' if i == 0 else f'prediction{i + 1}'
                    predictions[i] = [
                        str(_) for _ in predictions[i]
                    ]  # Fix the dictionary order to prevent the following situations: {'assistant':'', 'round':2, 'user':''}
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
            raise NotImplementedError(
                f'{predictions[0][0]} with type {type(predictions[0][0])}, please check the postprocess you add to the prediction string is right or not, we suggest to return an empty string but not None'
            )

        if self.dataset_cfg:
            dataset = build_dataset_from_cfg(self.dataset_cfg)
            if self.multi_eval:
                new_ds = {
                    k: dataset.test[k] * self.multi_eval_times
                    for k in dataset.test.column_names
                }
                dataset.reader.dataset['test'] = Dataset.from_dict(new_ds)
            if infer_order == 'double':
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
            dataset = LMEvalDataset(
                reader_cfg=dict(input_columns=input_columns,
                                output_column=None,
                                train_split='test'),
                reference=references,
                **pred_dict,
            )
        dataset.reader.output_column = 'reference'
        retriever = ZeroRetriever(dataset)

        if meta:
            self.inferencer.inference(
                retriever=retriever,
                prompt_template=self.meta_review_prompt_tmpl)
        else:
            self.inferencer.inference(retriever=retriever,
                                      prompt_template=self.prompt_tmpl)
        output = mmengine.load(self.output_path)
        return self.postprocess(output)

    def postprocess(self, output: Dict) -> Dict:
        """Postprocess output by adding necessary statistics or data into
        it."""
        if self.dict_postprocessor is None:
            return output
        else:
            kwargs = self.dict_postprocessor
            proc = DICT_POSTPROCESSORS.get(kwargs.pop('type'))
            return proc(output, self.output_path, **kwargs)
