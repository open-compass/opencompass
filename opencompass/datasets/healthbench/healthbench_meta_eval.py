"""This script evaluates a grader model on grading HealthBench rubrics. It
effectively evaluates the evaluator against physician opinion, so we call it a
meta-evaluation.

To run, use the following command (working directory should contain simple-
evals folder): `python -m simple-evals.simple_evals  --eval=healthbench_meta
--model=gpt-4.1`
"""

import json
import random
from collections import defaultdict
from typing import Literal

import blobfile as bf

from . import common
from .healthbench_eval import GRADER_TEMPLATE, parse_json_to_dict
from .types import Eval, EvalResult, SamplerBase, SingleEvalResult

INPUT_PATH = 'https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_meta_eval.jsonl'
INDEX_STR_TEMPLATE = 'pairwise_{model_or_physician}_{metric}_{pred_str}'
CLUSTER_STR_TEMPLATE = '{cluster}: {index_str}'

HEALTHBENCH_META_HTML_JINJA = (common.HTML_JINJA.replace(
    '<p>Correct Answer: {{ correct_answer }}</p>\n',
    '',
) + "<p>Explanation for grader's label: {{ explanation }}</p>")


class HealthBenchMetaEval(Eval):

    def __init__(
        self,
        grader_model: SamplerBase,
        num_examples: int | None = None,
        n_threads: int = 120,
        n_repeats: int = 1,
    ):
        with bf.BlobFile(INPUT_PATH, 'rb') as f:
            examples = [json.loads(line) for line in f]
        print(f'Loaded {len(examples)} examples from {INPUT_PATH}')

        rng = random.Random(0)

        if num_examples is not None and len(examples) > num_examples:
            examples = rng.sample(examples, num_examples)

        self.examples = examples * n_repeats
        self.grader_model = grader_model
        self.n_threads = n_threads

    def grade_sample(
        self,
        grading_response_dict: dict,
        physician_labels: list[bool],
        category: str,
    ) -> tuple[dict, bool | None, str]:
        metrics = {
            'num_physician_labels': len(physician_labels),
            'percent_physician_pos':
            sum(physician_labels) / len(physician_labels),
        }

        grader_label = grading_response_dict['criteria_met']
        assert grader_label is True or grader_label is False
        metrics['model_predicted_positive'] = grader_label
        explanation = grading_response_dict.get('explanation',
                                                'No explanation provided')

        category_metrics = {f'{category}: {k}': v for k, v in metrics.items()}
        metrics = {**metrics, **category_metrics}
        return metrics, grader_label, explanation

    def __call__(self, sampler: SamplerBase) -> EvalResult:

        def fn(row: dict) -> tuple[SingleEvalResult, bool | None]:
            convo_with_response = row['prompt'] + [
                dict(content=row['completion'], role='assistant')
            ]
            prompt_str = '\n\n'.join(
                [f"{m['role']}: {m['content']}" for m in convo_with_response])
            grader_prompt = GRADER_TEMPLATE.replace('<<conversation>>',
                                                    prompt_str)
            grader_prompt = grader_prompt.replace('<<rubric_item>>',
                                                  row['rubric'])
            grader_convo = [dict(content=grader_prompt, role='user')]

            while True:
                sampler_response = sampler(grader_convo)
                response_text = sampler_response.response_text
                actual_queried_grader_convo = (
                    sampler_response.actual_queried_message_list)
                grading_response_dict = parse_json_to_dict(response_text)
                if 'criteria_met' in grading_response_dict:
                    label = grading_response_dict['criteria_met']
                    if label is True or label is False:
                        break
                print('Grading failed due to bad JSON output, retrying...')

            metrics, grader_label, explanation = self.grade_sample(
                grading_response_dict=grading_response_dict,
                physician_labels=row['binary_labels'],
                category=row['category'],
            )
            score = metrics['model_predicted_positive']

            # Create HTML for each sample result
            html = common.jinja_env.from_string(
                HEALTHBENCH_META_HTML_JINJA).render(
                    prompt_messages=actual_queried_grader_convo,
                    next_message=dict(content=response_text, role='assistant'),
                    score=metrics['model_predicted_positive'],
                    extracted_answer=response_text,
                    explanation=explanation,
                )
            convo = actual_queried_grader_convo + [
                dict(content=response_text, role='assistant')
            ]
            return (
                SingleEvalResult(html=html,
                                 score=score,
                                 convo=convo,
                                 metrics=metrics),
                grader_label,
            )

        # Run evaluation and collect results
        all_outputs = common.map_with_progress(fn, self.examples,
                                               self.n_threads)
        results: list[SingleEvalResult]
        grader_labels: list[bool]
        results, grader_labels = zip(*all_outputs)

        # model pairwise agreement metrics
        model_agreement_metrics = compute_metrics_for_rater_by_class(
            self_pred_list=grader_labels,
            other_preds_list=[x['binary_labels'] for x in self.examples],
            cluster_list=[x['category'] for x in self.examples],
            model_or_physician='model',
        )

        # physicians:
        physician_rating_lists = defaultdict(lambda: ([], [], []))
        for example in self.examples:
            for i in range(len(example['binary_labels'])):
                physician_id = example['anonymized_physician_ids'][i]
                self_pred = example['binary_labels'][i]
                other_preds = (example['binary_labels'][:i] +
                               example['binary_labels'][i + 1:])
                cluster = example['category']
                physician_rating_lists[physician_id][0].append(self_pred)
                physician_rating_lists[physician_id][1].append(other_preds)
                physician_rating_lists[physician_id][2].append(cluster)

        physician_agreement_metric_lists = defaultdict(dict)
        for physician_id, (
                physician_rating_list,
                other_preds_list,
                cluster_list,
        ) in physician_rating_lists.items():
            physician_agreement_metrics = compute_metrics_for_rater_by_class(
                self_pred_list=physician_rating_list,
                other_preds_list=other_preds_list,
                cluster_list=cluster_list,
                model_or_physician='physician',
            )
            for k, v in physician_agreement_metrics.items():
                physician_agreement_metric_lists[k][physician_id] = v

        # consolidate final metrics and add agreement metrics
        final_metrics = common.aggregate_results(
            results, default_stats=('mean', 'n_samples', 'bootstrap_std'))
        model_agreement_metrics_condensed: dict[str, float] = {
            k: v['value']
            for k, v in model_agreement_metrics.items()
            if v['value'] is not None
        }
        assert final_metrics.metrics is not None
        final_metrics.metrics.update(model_agreement_metrics_condensed)
        final_metrics.score = final_metrics.metrics[
            'pairwise_model_f1_balanced']

        final_metrics.metadata = {
            'model_agreement_metrics': model_agreement_metrics,
            'physician_agreement_metric_lists':
            physician_agreement_metric_lists,
        }
        return final_metrics


def compute_metrics_for_rater_by_class(
    self_pred_list: list[bool],
    other_preds_list: list[list[bool]],
    cluster_list: list[str],
    model_or_physician: Literal['model', 'physician'],
) -> dict[str, dict[str, float | None]]:
    # get all the metrics for each cluster
    metric_lists = defaultdict(list)
    for self_pred, other_preds, cluster in zip(self_pred_list,
                                               other_preds_list,
                                               cluster_list,
                                               strict=True):
        self_pred_str = 'pos' if self_pred else 'neg'
        for other_pred in other_preds:
            # precision. based on the grader's labels -
            # i.e., calculated as TP / (TP + FP)
            # so a prediction should be recorded whenever self_pred is True
            precision_index_str = INDEX_STR_TEMPLATE.format(
                model_or_physician=model_or_physician,
                metric='precision',
                pred_str=self_pred_str,
            )
            metric_lists[precision_index_str].append(self_pred == other_pred)
            precision_cluster_str = CLUSTER_STR_TEMPLATE.format(
                cluster=cluster, index_str=precision_index_str)
            metric_lists[precision_cluster_str].append(self_pred == other_pred)

            # recall. based on the ground truth labels -
            # i.e., calculated as TP / (TP + FN)
            # so a prediction should be recorded whenever other_pred is True
            other_pred_str = 'pos' if other_pred else 'neg'
            recall_index_str = INDEX_STR_TEMPLATE.format(
                model_or_physician=model_or_physician,
                metric='recall',
                pred_str=other_pred_str,
            )
            metric_lists[recall_index_str].append(self_pred == other_pred)
            recall_cluster_str = CLUSTER_STR_TEMPLATE.format(
                cluster=cluster, index_str=recall_index_str)
            metric_lists[recall_cluster_str].append(self_pred == other_pred)

    metrics: dict[str, dict[str, float | None]] = {}
    for index_str, metric_list in metric_lists.items():
        n = len(metric_list)
        metric = sum(metric_list) / n if n > 0 else None
        metrics[index_str] = {
            'n': n,
            'value': metric,
        }

    f1_metrics = get_f1_metrics(metrics)
    metrics.update(f1_metrics)

    balanced_metrics = get_balanced_metrics(metrics)
    metrics.update(balanced_metrics)

    return metrics


def get_f1_metrics(
    metrics: dict[str, dict[str, float | None]],
) -> dict[str, dict[str, float | None]]:
    f1_metrics: dict[str, dict[str, float | None]] = {}
    for precision_key_name in metrics:
        if 'precision' in precision_key_name:
            recall_key_name = precision_key_name.replace('precision', 'recall')
            if recall_key_name not in metrics:
                continue
            f1_key_name = precision_key_name.replace('precision', 'f1')
            assert f1_key_name not in metrics
            f1_metrics[f1_key_name] = compute_f1_metric(
                precision=metrics[precision_key_name],
                recall=metrics[recall_key_name],
            )

    return f1_metrics


def compute_f1_metric(
    precision: dict[str, float | None],
    recall: dict[str, float | None],
) -> dict[str, float | None]:
    precision_n = precision['n']
    recall_n = recall['n']
    assert precision_n is not None and recall_n is not None, 'n_pos or n_neg is None'

    precision_metric = precision['value']
    recall_metric = recall['value']
    if precision_metric is None or recall_metric is None:
        f1_metric = None
        n_f1 = (
            precision_n + recall_n
        )  # precision_metric is None iff precision_n = 0 and recall_metric is None iff recall_n = 0, so if either is zero this gives TP + FN + FP without double counting
    elif precision_metric == 0 and recall_metric == 0:
        f1_metric = 0.0
        tp = precision_metric * precision_n  # because precision = TP / (TP+FP)
        n_f1 = precision_n + recall_n - tp  # TP+FP + TP+FN − TP
    else:
        f1_metric = (2 * (precision_metric * recall_metric) /
                     (precision_metric + recall_metric))
        tp = precision_metric * precision_n  # because precision = TP / (TP+FP)
        n_f1 = precision_n + recall_n - tp  # TP+FP + TP+FN − TP

    return {
        'n': n_f1,
        'value': f1_metric,
    }


def get_balanced_metrics(
    metrics: dict[str, dict[str, float | None]],
) -> dict[str, dict[str, float | None]]:
    balanced_metrics: dict[str, dict[str, float | None]] = {}
    for pos_key_name in metrics:
        if 'pos' in pos_key_name:
            neg_key_name = pos_key_name.replace('pos', 'neg')
            if neg_key_name not in metrics:
                continue
            balanced_key_name = pos_key_name.replace('pos', 'balanced')
            assert balanced_key_name not in metrics
            balanced_metrics[balanced_key_name] = compute_balanced_metric(
                metric_pos=metrics[pos_key_name],
                metric_neg=metrics[neg_key_name],
            )

    return balanced_metrics


def compute_balanced_metric(
    metric_pos: dict[str, float | None],
    metric_neg: dict[str, float | None],
) -> dict[str, float | None]:
    n_pos = metric_pos['n']
    n_neg = metric_neg['n']
    assert n_pos is not None and n_neg is not None, 'n_pos or n_neg is None'

    pos_metric = metric_pos['value']
    neg_metric = metric_neg['value']
    if pos_metric is None or neg_metric is None:
        metric = None
    else:
        metric = (pos_metric + neg_metric) / 2

    return {
        'n': n_pos + n_neg,
        # note: this overcounts samples going towards the balanced F1
        'value': metric,
    }
