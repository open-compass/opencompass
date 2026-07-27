import json
from numbers import Number
from pathlib import Path

import pandas as pd

from opencompass.datasets.vlmevalkit.builder import (_lmu_data_root,
                                                     build_vlmeval_dataset)
from opencompass.datasets.vlmevalkit.dataset import make_vlmeval_sample_id
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS


def _flatten_vlmeval_metrics(raw_metrics):
    from vlmeval.smp.status_report import flatten_summary_metrics

    return flatten_summary_metrics(raw_metrics)


@ICL_EVALUATORS.register_module()
class VLMEvalKitEvaluator(BaseEvaluator):

    def __init__(self,
                 dataset_name='MMBench_DEV_EN',
                 data_root=None,
                 dataset_kwargs=None,
                 eval_kwargs=None,
                 sample_limit=None,
                 pred_postprocessor=None):
        super().__init__(pred_postprocessor=pred_postprocessor)
        if sample_limit is not None and sample_limit <= 0:
            raise ValueError('VLMEvalKit sample_limit must be positive.')
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.dataset_kwargs = dict(dataset_kwargs or {})
        self.eval_kwargs = dict(eval_kwargs or {})
        self.sample_limit = sample_limit

    def score(self, predictions, test_set):
        if len(predictions) != len(test_set):
            raise ValueError(
                'VLMEvalKit predictions and test set have different lengths.')
        if any(prediction is None or not str(prediction).strip()
               for prediction in predictions):
            raise ValueError('Missing VLMEvalKit prediction.')

        sample_id = list(test_set['sample_id'])
        vlmeval_index = list(test_set['vlmeval_index'])
        dataset_name = list(test_set['dataset_name'])
        if len(sample_id) != len(set(sample_id)):
            raise ValueError('Duplicate VLMEvalKit sample_id in predictions.')
        wrong_names = set(dataset_name) - {self.dataset_name}
        if wrong_names:
            raise ValueError(
                f'Unexpected VLMEvalKit dataset names: {sorted(wrong_names)}')

        dataset = build_vlmeval_dataset(self.dataset_name, self.data_root,
                                        **self.dataset_kwargs)
        frame = dataset.data.copy()
        if self.sample_limit is not None:
            frame = frame.iloc[:self.sample_limit].copy()
        frame['_opencompass_sample_id'] = [
            make_vlmeval_sample_id(self.dataset_name, index)
            for index in frame['index']
        ]
        if len(frame['_opencompass_sample_id'].unique()) != len(frame):
            raise ValueError('Duplicate index in VLMEvalKit dataset.')

        expected = dict(
            zip(frame['_opencompass_sample_id'],
                [str(index) for index in frame['index']]))
        actual_ids = set(sample_id)
        unknown = actual_ids - set(expected)
        if unknown:
            raise ValueError(
                f'Unknown VLMEvalKit sample_id: {sorted(unknown)}')

        predictions_by_id = {}
        for current_id, index, prediction in zip(sample_id, vlmeval_index,
                                                 predictions):
            if str(index) != expected[current_id]:
                raise ValueError(
                    f'VLMEvalKit index mismatch for {current_id}: {index}')
            predictions_by_id[current_id] = str(prediction)

        frame = frame[frame['_opencompass_sample_id'].isin(sample_id)].copy()
        frame['prediction'] = [
            predictions_by_id[sample_id]
            for sample_id in frame['_opencompass_sample_id']
        ]
        frame = frame.drop(columns='_opencompass_sample_id')
        if 'image' in frame:
            frame = frame.drop(columns='image')

        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result_file = output_dir / f'{self.dataset_name}.xlsx'
        frame.to_excel(result_file, index=False)
        config_file = output_dir / 'vlmevalkit_evaluation.json'
        with config_file.open('w', encoding='utf-8') as f:
            json.dump(dict(dataset_name=self.dataset_name,
                           data_root=self.data_root,
                           dataset_kwargs=self.dataset_kwargs,
                           sample_limit=self.sample_limit,
                           eval_kwargs=self.eval_kwargs,
                           result_file=str(result_file)),
                      f,
                      indent=2,
                      ensure_ascii=False)

        with _lmu_data_root(self.data_root):
            raw_metrics = dataset.evaluate(str(result_file),
                                           **self.eval_kwargs)
        metrics, flattened, primary = self._convert_metrics(
            dataset, raw_metrics)
        metrics_file = output_dir / 'vlmevalkit_metrics.json'
        with metrics_file.open('w', encoding='utf-8') as f:
            json.dump(dict(flattened=flattened, primary=primary),
                      f,
                      indent=2,
                      ensure_ascii=False)
        return metrics

    @staticmethod
    def _convert_metrics(dataset, raw_metrics):
        flattened = _flatten_vlmeval_metrics(raw_metrics)
        primary = dataset.report_primary_metric(flattened)
        metrics = {}
        for key, value in flattened.items():
            if isinstance(value, Number) and not pd.isna(value):
                metrics[str(key)] = float(value) * 100
        for key, value in primary.items():
            if not isinstance(value, Number) or pd.isna(value):
                continue
            value = float(value)
            if key in flattened and value == float(flattened[key]):
                value *= 100
            metrics[str(key)] = value
        if not metrics:
            raise ValueError(
                'VLMEvalKit evaluate() returned no numeric metrics.')
        return metrics, flattened, primary
