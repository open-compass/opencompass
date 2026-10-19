from collections import defaultdict
from mmengine import load

from ..utils.template import parse_string
from ..utils.format_load import format_load
from ..schema import ResponseDataSample
import ast
import numpy as np

class InstructEvaluator:
    """Instruct Following Evaluation

    Args:
        dataset_path(str): File path of evaluation dataset.

    """

    def __init__(
        self,
        dataset_path: str,
        **kwargs,
    ) -> None:
        self.dataset_path = dataset_path

    def _load_dataset(self):
        self.dataset = []
        dataset = load(self.dataset_path)

        for key in dataset.keys():
            datum = dataset[key]
            data_sample = self._process_response(datum)

            self.dataset.append(
                dict(
                    origin_prompt=datum["origin_prompt"],
                    response_data_sample=data_sample))
        self.num_samples = len(self.dataset)

    def _process_response(
        self,
        datum: dict,
    ) -> ResponseDataSample:
        """Process the response to needed format.

        Args:
            datum(dict): inputs.

        Returns:
            dict: Processed response data sample.
        """

        # Dict with keyword-only arguments.
        template = datum['template']
        # Generated response.
        pred_data = datum['prediction']
        # Response of ground truth.
        gt_data = datum['ground_truth']
        meta_data = datum['meta_data']

        return ResponseDataSample(
            template=template, pred=pred_data, gt=gt_data, meta_data=meta_data)

    def _evaluate(self, data_sample: dict) -> dict:
        metrics_result = dict()
        response_format = data_sample.meta_data['response_format']
        if response_format == 'json':
            pred_data = self.json_format_parse(data_sample)
        else:
            pred_data = self.string_format_parse(data_sample)

        if pred_data is None:
            # directly set to 0 for all metrics
            metrics_result[f'{response_format}_format_metric'] = 0
            metrics_result[f'{response_format}_args_em_metric'] = 0
            return metrics_result

        # Exact matching
        metrics_result[f'{response_format}_format_metric'] = 1
        metrics_result[f'{response_format}_args_em_metric'] = self.compute_args_em_metric(
            gt_action=data_sample.gt['action'], pred_action=pred_data['action'],
            gt_args=data_sample.gt['args'], pred_args=pred_data['args']
        )
        return metrics_result

    def compute_args_em_metric(self, gt_action, pred_action, gt_args, pred_args):
        cnt = 0.
        if gt_action == pred_action:
            cnt += 1.
        num_args = len(gt_args) + 1     # 1 means action name match
        for gt_key in gt_args:
            pred_val = pred_args.get(gt_key, "")
            if pred_val == gt_args[gt_key]:
                cnt += 1.
        return cnt / num_args

    def string_format_parse(self, data_sample):
        pred_data = data_sample.pred
        template = data_sample.template
        thought_start = template['thought_start']
        thought_end = template['thought_end']
        action_start = template['action_start']
        action_end = template['action_end']
        args_start = template['args_start']
        args_end = template['args_end']

        parse_template = thought_start + "{thought}" + thought_end \
            + action_start + "{action}" + action_end \
            + args_start + "{args}" + args_end
        res = parse_string(parse_template, pred_data, allow_newline=True)
        try:
            if res is not None:
                args = ast.literal_eval(res['args'].strip())
                res['args'] = args if isinstance(args, dict) else {}
                res['action'] = res['action'].strip()
            return res
        except:
            return dict(thought=res['thought'], action=res['action'].strip(), args=dict())

    def json_format_parse(self, data_sample):
        try:
            pred_data = format_load(data_sample.pred)
            template = data_sample.template
            new_data = dict()
            new_data['thought'] = pred_data[template['thought']]
            new_data['action'] = pred_data[template['action']]
            args = pred_data[template['args']]
            new_data['args'] = args if isinstance(args, dict) else {}
        except Exception as e:
            return None

        return new_data

    def evaluate(self):
        self._load_dataset()
        results_list = []
        for data_sample in self.dataset:
            metrics_result = self._evaluate(data_sample['response_data_sample'])
            results_list.append(metrics_result)
        return self._post_process(results_list)

    def _post_process(self, results_list):
        # list of dict to dict of list
        results_dict = defaultdict(list)
        {
            results_dict[key].append(sub[key])
            for sub in results_list for key in sub
        }
        metric_list = ['json_format_metric', 'json_args_em_metric',
                       'string_format_metric', 'string_args_em_metric']
        for metric in metric_list:
            results_dict[metric] = np.round(np.mean(results_dict[metric]), decimals=4)
        return results_dict
