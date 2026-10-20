import json
from numpy import mean
from mmengine import load
import numpy as np
import json
import re
from tqdm import tqdm

from ..schema import ResponseDataSample
from ..utils.format_load import format_load
from sentence_transformers import SentenceTransformer, util


def input_postprocess(text: str) -> str:
    if isinstance(text, str):
        text = text.split('<|')[0]
        text = text.split('<eoa>\n')[0]
        text = text.split('<TOKENS_UNUSED_1>\n')[0]
        text = text.split('<|im_end|>')[0]
        if len(text) > 1 and text[:2] == '{{' and text[-2:] == '}}':
            text = text[1:-1]
        while len(text) > 0 and text[-1] == '\n':
            text = text[:-1]
    return str(text)

class ReasonRetrieveUnderstandEvaluator:
    """Planning Evaluation
    Args:
        dataset_path(str): File path of evaluation dataset
        bert_score_model(str): the bert_score model for sentence similarity, default = "all-mpnet-base-v2".
            Refer to https://www.sbert.net/docs/pretrained_models.html for more models.
    """
    def __init__(
        self,
        dataset_path: str,
        bert_score_model: str = "all-mpnet-base-v2", # ['thenlper/gte-large-zh', 'all-mpnet-base-v2']
        default_prompt_type: str = 'json',
        eval_type: str = 'reason',
        **kwargs,
    ) -> None:
        self.bert_score_model = bert_score_model
        print(bert_score_model)
        self.dataset_path = dataset_path
        # self.bertscore = evaluate.load('bertscore')
        self.default_prompt_type = default_prompt_type # ["json", "str"]
        self.eval_type = eval_type
        self.valid_data_count = None
        self.sentence_model = SentenceTransformer(self.bert_score_model)

    def _load_dataset(self):
        self.dataset = []
        dataset = load(self.dataset_path)
        total_error = 0
        total_count = 0
        for key in dataset.keys():
            datum = dataset[key]
            data_sample, error = self._process_response(datum)
            total_error += error
            total_count += 1
            self.dataset.append(
                dict(response_data_sample=data_sample))

        self.num_samples = len(self.dataset)
        # print("total_data_count:", total_count, "valid_data_count:", total_count - total_error)
        self.valid_data_count = total_count - total_error

    def format_load(self, data):
        r'''
            ensure evaluator can work correctly under any data input
        '''
        try:
            json_format = format_load(data, start_character='{', end_character='}')
        except Exception as e:
            return {}
        if type(json_format) != dict:
            return {}
        prepared_json_format = dict()
        try:
            prepared_json_format['thought'] = str(json_format['thought'])
        except Exception as e:
            prepared_json_format['thought'] = ''
        try:
            prepared_json_format['name'] = str(json_format['name'])
        except Exception as e:
            prepared_json_format['name'] = ''

        if self.default_prompt_type == 'json':
            try:
                if isinstance(json_format['args'], dict):
                    prepared_json_format['args'] = json_format['args']
                else:
                    prepared_json_format['args'] = dict()
            except:
                prepared_json_format['args'] = dict()
        else:
            try:
                prepared_json_format['args'] = str(json_format['args'])
            except Exception as e:
                prepared_json_format['args'] = ""

        return prepared_json_format

    def _process_response(
        self,
        datum,
    ) -> ResponseDataSample:
        """Process the response to needed format.
        Args:
            datum(dict): inputs.
        Returns:
            dict: Processed response data sample.
        """

        # Generated response, which can be a string or list
        pred_data = datum['prediction']
        # Response of ground truth, which can be a string or list
        gt_data = datum['ground_truth']
        # prompt_type: The type of planning prompt, supporting "json" and "ReWOO"
        if "meta_data" in datum:
            prompt_type = datum["meta_data"].get("response_format", self.default_prompt_type)
        else:
            prompt_type = self.default_prompt_type

        error = 0
        gt = self.format_load(gt_data)
        # pred_data = input_postprocess(pred_data)

        if prompt_type == 'json':
            pred = self.format_load(pred_data)
            if pred == {} or gt == {}:
                error = 1
        elif prompt_type == 'str':
            # choose the first line
            pred = dict()
            if self.eval_type == 'reason':
                pred['thought'] = pred_data
            if self.eval_type == 'retrieve':
                pred['name'] = pred_data
            if self.eval_type == 'understand':
                pred['args'] = pred_data
        else:
            raise NotImplementedError(f"Currently, we only support json and str format, but get {prompt_type}")

        if error == 1:
            pred = dict()
        return ResponseDataSample(template = '', pred=pred, gt=gt), error

    def _evaluate(self, data_sample):
        """Evaluate the response data sample.
        """
        # To enable batch evaluation, the evaluator is written at post_process.
        return data_sample

    def evaluate(self):
        self._load_dataset()
        results_list = []
        for data_sample in tqdm(self.dataset):
            metrics_result = self._evaluate(
                data_sample['response_data_sample'])
            results_list.append(metrics_result)
        return self._post_process(results_list)

    def find_a_dot_b_structure(self, text):
        # find a.b structure
        pattern = r'\w+\.\w+'
        return re.findall(pattern, text)

    def find_FinishAction(self, text):
        # find FinishAction
        pattern = r'FinishAction'
        return re.findall(pattern, text)

    def _post_process(self, results_list):
        # list of dict to dict of list
        if self.default_prompt_type == 'json':
            metric_keys = ['thought', 'name', 'args', 'parse_rate']
        if self.default_prompt_type == 'str':
            if self.eval_type == 'reason':
                metric_keys = ['thought', 'parse_rate']
            if self.eval_type == 'retrieve':
                metric_keys = ['name', 'parse_rate']
            if self.eval_type == 'understand':
                metric_keys = ['args', 'parse_rate']
        metrics_results = []
        batch_data = []; batch_arg_data = []
        batch_id = []; batch_arg_id = []
        BATCH_LIMIT = 32
        for id, data in enumerate(results_list):
            metrics_results.append(
                {metric_keys[x]: 0 for x in range(len(metric_keys))}
            )
            if len(data.pred.keys()) != 0:
                metrics_results[id]['parse_rate'] = 1
            if 'thought' in data.pred and 'thought' in data.gt:
                batch_data.extend([data.pred['thought'], data.gt['thought']])
                batch_id.extend([id])
                if len(batch_data) >= BATCH_LIMIT:
                    pred_emb = self.sentence_model.encode(batch_data, convert_to_tensor=True)
                    for i in range(0, len(batch_data), 2):
                        cosine_score = np.maximum(util.cos_sim(pred_emb[i], pred_emb[i+1]).cpu().numpy(), 0)
                        metrics_results[batch_id[i // 2]]['thought'] = cosine_score[0, 0]
                    batch_data = []
                    batch_id = []
            if 'name' in data.pred and 'name' in data.gt:
                if self.default_prompt_type == 'json':
                    if data.pred['name'] == data.gt['name']:
                        metrics_results[id]['name'] = 1
                    else:
                        metrics_results[id]['name'] = 0
                else:
                    if data.gt['name'] not in data.pred['name']:
                        metrics_results[id]['name'] = 0
                    else:
                        metrics_results[id]['name'] = 1
                        find_all_name = self.find_a_dot_b_structure(data.pred['name']) + self.find_FinishAction(data.pred['name'])
                        for name in find_all_name:
                            if name != data.gt['name']:
                                metrics_results[id]['name'] = 0

            if 'args' in data.pred and 'args' in data.gt:
                batch_arg_data.extend([str(data.pred['args']), str(data.gt['args'])])
                batch_arg_id.extend([id])
                if len(batch_arg_data) >= BATCH_LIMIT:
                    pred_emb = self.sentence_model.encode(batch_arg_data, convert_to_tensor=True)
                    for i in range(0, len(batch_arg_data), 2):
                        cosine_score = np.maximum(util.cos_sim(pred_emb[i], pred_emb[i+1]).cpu().numpy(), 0)
                        metrics_results[batch_arg_id[i // 2]]['args'] = cosine_score[0, 0]
                    batch_arg_data = []
                    batch_arg_id = []

        if len(batch_data) > 0:
            pred_emb = self.sentence_model.encode(batch_data, convert_to_tensor=True)
            for i in range(0, len(batch_data), 2):
                cosine_score = np.maximum(util.cos_sim(pred_emb[i], pred_emb[i+1]).cpu().numpy(), 0)
                metrics_results[batch_id[i // 2]]['thought'] = cosine_score[0, 0]
            batch_data = []
            batch_id = []

        if len(batch_arg_data) > 0:
            pred_emb = self.sentence_model.encode(batch_arg_data, convert_to_tensor=True)
            for i in range(0, len(batch_arg_data), 2):
                cosine_score = np.maximum(util.cos_sim(pred_emb[i], pred_emb[i+1]).cpu().numpy(), 0)
                metrics_results[batch_arg_id[i // 2]]['args'] = cosine_score[0, 0]
            batch_arg_data = []
            batch_arg_id = []

        results = dict()
        for key in metric_keys:
            results[key] = mean([metrics_results[key] for metrics_results in metrics_results])
        return results

class ReasonRetrieveUnderstandEvaluatorNoBatch:
    """Planning Evaluation
    Args:
        dataset_path(str): File path of evaluation dataset
        bert_score_model(str): the bert_score model for sentence similarity, default = "all-mpnet-base-v2".
            Refer to https://www.sbert.net/docs/pretrained_models.html for more models.
    """
    def __init__(
        self,
        dataset_path: str,
        bert_score_model: str = "all-mpnet-base-v2",
        default_prompt_type: str = 'json',
        eval_type: str = 'reason',
    ) -> None:
        self.bert_score_model = bert_score_model
        self.dataset_path = dataset_path
        # self.bertscore = evaluate.load('bertscore')
        self.default_prompt_type = default_prompt_type # ["json", "str"]
        self.eval_type = eval_type
        self.valid_data_count = None
        self.sentence_model = SentenceTransformer(self.bert_score_model)

    def _load_dataset(self):
        self.dataset = []
        dataset = load(self.dataset_path)
        total_error = 0
        total_count = 0
        for key in dataset.keys():
            datum = dataset[key]
            data_sample, error = self._process_response(datum)
            total_error += error
            total_count += 1
            self.dataset.append(
                dict(response_data_sample=data_sample))

        self.num_samples = len(self.dataset)
        # print("total_data_count:", total_count, "valid_data_count:", total_count - total_error)
        self.valid_data_count = total_count - total_error

    def format_load(self, data):
        r'''
            ensure evaluator can work correctly under any data input
        '''
        if type(data) == dict:
            json_format = data
        else:
            try:
                json_format = json.loads(data) #json.loads(pred_data)
            except Exception as e:
                return {}
        if type(json_format) != dict:
            return {}
        prepared_json_format = dict()
        try:
            prepared_json_format['thought'] = str(json_format['thought'])
        except Exception as e:
            prepared_json_format['thought'] = ''
        try:
            prepared_json_format['name'] = str(json_format['name'])
        except Exception as e:
            prepared_json_format['name'] = ''
        try:
            if prepared_json_format["name"] != "FinishAction":
                arg_inputs = json_format["args"]
                if type(arg_inputs) == str:
                    arg_inputs = json.loads(arg_inputs)
                if type(arg_inputs) == dict:
                    prepared_json_format['args'] = arg_inputs
                else:
                    prepared_json_format["args"] = {}
            else:
                prepared_json_format["args"] = {}
        except Exception as e:
            prepared_json_format['args'] = {}
        return prepared_json_format

    def _process_response(
        self,
        datum,
    ) -> ResponseDataSample:
        """Process the response to needed format.
        Args:
            datum(dict): inputs.
        Returns:
            dict: Processed response data sample.
        """

        # Generated response, which can be a string or list
        pred_data = datum['prediction']
        # Response of ground truth, which can be a string or list
        gt_data = datum['ground_truth']
        # prompt_type: The type of planning prompt, supporting "json" and "ReWOO"
        if "meta" in datum:
            prompt_type = datum["meta"].get("response_format", self.default_prompt_type)
        else:
            prompt_type = self.default_prompt_type

        error = 0
        gt = self.format_load(gt_data)
        # pred_data = input_postprocess(pred_data)
        if prompt_type == 'json':
            # pred_data = pred_data.replace('\'', '\"')
            pred = self.format_load(pred_data)
            if pred == {} or gt == {}:
                error = 1
        elif prompt_type == 'str':
            # choose the first line
            pred = dict()
            if self.eval_type == 'reason':
                pred['thought'] = pred_data
            if self.eval_type == 'retrieve':
                pred['name'] = pred_data
            if self.eval_type == 'understand':
                # pred_data = pred_data.replace('\'', '\"')
                # try:
                #     pred['args'] = json.loads(pred_data)
                #     if type(pred['args']) != dict:
                #         pred['args'] = {}
                # except Exception as e:
                #     error = 1
                pred['args'] = pred_data
        else:
            raise NotImplementedError(f"Currently, we only support json and str format, but get {prompt_type}")

        if error == 1:
            pred = dict()
        return ResponseDataSample(template = '', pred=pred, gt=gt), error

    def _evaluate(self, data_sample) -> dict:
        """Evaluate the response data sample.
        """
        metrics_result = {
            'thought': 0,
            'name': 0,
            'args_precision': 0,
            'args_recall': 0,
            'args_f1_score': 0,
            'parse_rate': 0,
        }
        if 'thought' in data_sample.pred and 'thought' in data_sample.gt:
            pred_emb = self.sentence_model.encode(data_sample.pred['thought'], convert_to_tensor=True)
            gt_emb = self.sentence_model.encode(data_sample.gt['thought'], convert_to_tensor=True)
            cosine_scores = np.maximum(util.cos_sim(pred_emb, gt_emb).cpu().numpy(), 0)
            metrics_result['thought'] = cosine_scores[0, 0]

        if 'name' in data_sample.pred and 'name' in data_sample.gt:
            if data_sample.pred['name'] == data_sample.gt['name']:
                metrics_result['name'] = 1
            else:
                metrics_result['name'] = 0
        if 'args' in data_sample.pred and 'args' in data_sample.gt:
            gt_num_keys = len(data_sample.gt['args'].keys())
            pred_num_keys = len(data_sample.pred['args'].keys())
            if pred_num_keys == 0 and gt_num_keys == 0:
                metrics_result['args_precision'] = 1
                metrics_result['args_recall'] = 1
                metrics_result['args_f1_score'] = 1
            elif pred_num_keys == 0 or gt_num_keys == 0:
                metrics_result['args_precision'] = 0
                metrics_result['args_recall'] = 0
                metrics_result['args_f1_score'] = 0
            else:
                correct_count = 0
                for key in data_sample.gt['args'].keys():
                    if key in data_sample.pred['args'] and str(data_sample.pred['args'][key]) == str(data_sample.gt['args'][key]):
                        correct_count += 1
                metrics_result['args_precision'] = correct_count / pred_num_keys
                metrics_result['args_recall'] = correct_count / gt_num_keys
                if metrics_result['args_precision'] + metrics_result['args_recall'] == 0:
                    metrics_result['args_f1_score'] = 0
                else:
                    metrics_result['args_f1_score'] = 2 * metrics_result['args_precision'] * metrics_result['args_recall'] / \
                    (metrics_result['args_precision'] + metrics_result['args_recall'])

        if len(data_sample.pred.keys()) == 0:
            metrics_result['parse_rate'] = 0
        else:
            metrics_result['parse_rate'] = 1
        return metrics_result

    def evaluate(self):
        self._load_dataset()
        results_list = []
        for data_sample in tqdm(self.dataset):
            metrics_result = self._evaluate(
                data_sample['response_data_sample'])
            results_list.append(metrics_result)
        return self._post_process(results_list)

    def _post_process(self, results_list):
        # list of dict to dict of list
        results = dict()
        if self.default_prompt_type == 'json':
            metric_keys = ['thought', 'name', 'args_precision', 'args_recall', 'args_f1_score', 'parse_rate']
        if self.default_prompt_type == 'str':
            if self.eval_type == 'reason':
                metric_keys = ['thought', 'parse_rate']
            if self.eval_type == 'retrieve':
                metric_keys = ['name', 'parse_rate']
            if self.eval_type == 'understand':
                metric_keys = ['args_precision', 'args_recall', 'args_f1_score', 'parse_rate']
        for key in metric_keys:
            results[key] = mean([result[key] for result in results_list])
        return results
