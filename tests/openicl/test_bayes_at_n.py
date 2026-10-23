import json
import math

import pytest
from datasets import Dataset
from mmengine.config import ConfigDict

from opencompass.datasets import CustomDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator, EMEvaluator
from opencompass.summarizers.default import DefaultSummarizer
from opencompass.tasks.openicl_eval import OpenICLEvalTask
from opencompass.utils import get_logger


def repeated_dataset(size, n, subdivisions=None):
    subdivisions = subdivisions or ['math_easy'] * size
    return Dataset.from_list([{
        'subdivision': subdivisions[index],
        'idx': index,
        'answer': ['yes'],
    } for _ in range(n) for index in range(size)])


class BinaryEvaluator(BaseEvaluator):

    def score(self, predictions, references=None):
        return {
            'accuracy': 100.0 * sum(predictions) / len(predictions),
            'details': [{
                'correct': value
            } for value in predictions],
        }


@pytest.mark.parametrize('predictions,variance', [
    ([0, 1, 0, 1], 3 / 160),
    ([0, 0, 1, 1], 1 / 40),
    ([0, 0, 0, 1], 7 / 320),
])
def test_independent_beta_moments(predictions, variance):
    result = BinaryEvaluator().evaluate(k=1,
                                        n=2,
                                        original_dataset=repeated_dataset(
                                            2, 2),
                                        bayes_at_n=True,
                                        predictions=predictions)
    expected_mean = (sum(predictions) + 2) / 8
    assert result['Bayes@2'] == pytest.approx(100 * expected_mean)
    assert result['Bayes@2 posterior SD'] == pytest.approx(100 *
                                                           math.sqrt(variance))
    assert result['math_easy/Bayes@2 posterior SD'] == result[
        'Bayes@2 posterior SD']


@pytest.mark.parametrize('outcome', [0, 1])
def test_one_trial_keeps_prior_uncertainty(outcome):
    result = BinaryEvaluator().evaluate(k=1,
                                        n=1,
                                        original_dataset=repeated_dataset(
                                            1, 1),
                                        bayes_at_n=True,
                                        predictions=[outcome])
    assert result['Bayes@1'] == pytest.approx(100 * (1 + outcome) / 3)
    assert result['Bayes@1 posterior SD'] == pytest.approx(100 / math.sqrt(18))


def test_subdivision_moments_and_reordered_trials():
    dataset = repeated_dataset(3, 2, ['a_b', 'a_b', 'c'])
    predictions = [0, 1, 1, 1, 1, 0]
    evaluator = BinaryEvaluator()
    result = evaluator.evaluate(1,
                                2,
                                dataset,
                                bayes_at_n=True,
                                predictions=predictions)
    assert result['a_b/Bayes@2'] == pytest.approx(62.5)
    assert result['a_b/Bayes@2 posterior SD'] == pytest.approx(
        100 * math.sqrt(7 / 320))
    assert result['Bayes@2 posterior SD'] == pytest.approx(
        100 * math.sqrt(11 / 80) / 3)
    order = [2, 0, 1, 4, 5, 3]
    reordered = evaluator.evaluate(1,
                                   2,
                                   dataset.select(order),
                                   bayes_at_n=True,
                                   predictions=[predictions[i] for i in order])
    for metric in ('Bayes@2', 'Bayes@2 posterior SD', 'a_b/Bayes@2'):
        assert reordered[metric] == result[metric]


@pytest.mark.parametrize('value', [None, 0.25, '1', float('nan'), 2])
def test_reject_nonbinary_or_missing_correctness(value):

    class BadEvaluator(BaseEvaluator):

        def score(self, predictions):
            return {'details': [{'correct': value}]}

    with pytest.raises(ValueError, match='binary correctness'):
        BadEvaluator().evaluate(1,
                                1,
                                repeated_dataset(1, 1),
                                bayes_at_n=True,
                                predictions=['x'])


@pytest.mark.parametrize('scored', [
    {},
    {
        'details': []
    },
    {
        'error': 'failed',
        'details': [{
            'correct': True
        }]
    },
    {
        'details': [{
            'error': 'timeout',
            'correct': False
        }]
    },
])
def test_reject_incomplete_or_failed_scoring(scored):

    class BadEvaluator(BaseEvaluator):

        def score(self, predictions):
            return dict(scored)

    with pytest.raises(ValueError, match='Bayes@N'):
        BadEvaluator().evaluate(1,
                                1,
                                repeated_dataset(1, 1),
                                bayes_at_n=True,
                                predictions=['x'])


@pytest.mark.parametrize('n,dataset,predictions', [
    (0, repeated_dataset(1, 1), [1]),
    (True, repeated_dataset(1, 1), [1]),
    (2, repeated_dataset(1, 1), [1]),
    (1, Dataset.from_list([]), []),
    (2, repeated_dataset(2, 2).select([0, 0, 2, 3]), [1] * 4),
    (2, repeated_dataset(2, 2), [1] * 3),
    (1, Dataset.from_list([{
        'idx': 0
    }]), [1]),
])
def test_reject_invalid_repeated_inputs(n, dataset, predictions):
    with pytest.raises(ValueError, match='Bayes@N'):
        BinaryEvaluator().evaluate(1,
                                   n,
                                   dataset,
                                   bayes_at_n=True,
                                   predictions=predictions)


def test_positional_detail_dictionary_order():

    class DictEvaluator(BaseEvaluator):

        def score(self, predictions):
            return {
                'details': {
                    '1': {
                        'is_correct': True
                    },
                    '0': {
                        'is_correct': False
                    }
                }
            }

    result = DictEvaluator().evaluate(1,
                                      1,
                                      repeated_dataset(2, 1, ['a', 'b']),
                                      bayes_at_n=True,
                                      predictions=['x', 'x'])
    assert result['a/Bayes@1'] == pytest.approx(100 / 3)
    assert result['b/Bayes@1'] == pytest.approx(200 / 3)


def test_missing_details_in_one_replica_fail_before_grouping():

    class PartialEvaluator(BinaryEvaluator):

        def score(self, predictions):
            result = super().score(predictions)
            if self.dataset_replica_idx == 1:
                result.pop('details')
            return result

    with pytest.raises(ValueError, match='every replica'):
        PartialEvaluator().evaluate(1,
                                    2,
                                    repeated_dataset(1, 2),
                                    bayes_at_n=True,
                                    predictions=[0, 1])


def test_bayes_moments_do_not_depend_on_k():
    evaluator = BinaryEvaluator()
    kwargs = dict(n=2,
                  original_dataset=repeated_dataset(2, 2),
                  bayes_at_n=True,
                  predictions=[0, 1, 0, 1])
    result = evaluator.evaluate(k=[1, 2], **kwargs)
    assert result['Bayes@2 posterior SD'] == pytest.approx(100 *
                                                           math.sqrt(3 / 160))
    assert 'G-Pass@2_0.0' in result


def test_disabled_output_is_unchanged():
    result = BinaryEvaluator().evaluate(1,
                                        2,
                                        repeated_dataset(2, 2),
                                        predictions=[0, 1, 0, 1])
    assert result['accuracy (2 runs average)'] == 50
    assert all('Bayes@' not in key for key in result)


def test_registered_evaluator_task_json_and_summary(tmp_path):
    task = OpenICLEvalTask.__new__(OpenICLEvalTask)
    task.eval_cfg = ConfigDict(evaluator=dict(type='EMEvaluator'),
                               bayes_at_n=True)
    task.dataset_cfg = ConfigDict(abbr='questions', n=2, k=1)
    task.model_cfg = ConfigDict(abbr='model')
    task.work_dir = str(tmp_path)
    task.output_column = 'answer'
    task.dump_details = False
    task.logger = get_logger()
    source = tmp_path / 'questions.jsonl'
    source.write_text('\n'.join(
        json.dumps({
            'question': f'q{i}',
            'answer': ['yes']
        }) for i in range(2)))
    dataset = CustomDataset(path=str(source),
                            n=2,
                            reader_cfg=dict(input_columns=['question'],
                                            output_column='answer'))
    assert len(dataset.test) == 4
    predictions = ['no', 'yes', 'no', 'yes']
    result = task._evaluate_predictions(predictions, dataset.test,
                                        [{
                                            'prediction': prediction
                                        } for prediction in predictions])
    assert result['Bayes@2'] == 50
    assert result['Bayes@2 posterior SD'] == pytest.approx(100 *
                                                           math.sqrt(3 / 160))
    assert 'details' not in result
    path = tmp_path / 'results' / 'model' / 'questions.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(result))
    summarizer = DefaultSummarizer(
        ConfigDict(models=[task.model_cfg],
                   datasets=[task.dataset_cfg],
                   work_dir=str(tmp_path)))
    _, parsed, metrics, _ = summarizer._pick_up_results()
    assert parsed['model']['questions']['Bayes@2'] == 50
    assert 'Bayes@2 posterior SD' in metrics['questions']


def test_exact_match_normalization_is_preserved():
    result = EMEvaluator().evaluate(1,
                                    1,
                                    repeated_dataset(1, 1),
                                    bayes_at_n=True,
                                    predictions=['  yes! '],
                                    references=[['yes']])
    assert result['Bayes@1'] == pytest.approx(200 / 3)


@pytest.mark.parametrize('explicit', [False, True])
def test_summary_groups_do_not_average_posteriors(tmp_path, explicit):
    group = dict(
        name='combined',
        subsets=([('a', 'Bayes@2 posterior SD'),
                  ('b', 'Bayes@2 posterior SD')] if explicit else ['a', 'b']))
    summarizer = DefaultSummarizer(ConfigDict(models=[dict(abbr='model')],
                                              datasets=[],
                                              work_dir=str(tmp_path)),
                                   summary_groups=[group])
    parsed = {
        'model': {
            name: {
                'accuracy': 50,
                'Bayes@2': 50,
                'Bayes@2 posterior SD': 10
            }
            for name in ('a', 'b')
        }
    }
    metrics = {
        name: ['accuracy', 'Bayes@2', 'Bayes@2 posterior SD']
        for name in ('a', 'b')
    }
    if explicit:
        with pytest.raises(ValueError, match='cross-dataset'):
            summarizer._calculate_group_metrics({}, parsed, metrics, {})
    else:
        _, combined, _, _ = summarizer._calculate_group_metrics({'model': {}},
                                                                parsed,
                                                                metrics, {
                                                                    'a': 'gen',
                                                                    'b': 'gen'
                                                                })
        assert combined['model']['combined'] == {
            'accuracy': 50,
            'naive_average': 50
        }


def test_summary_group_cannot_name_accuracy_as_posterior_sd(tmp_path):
    summarizer = DefaultSummarizer(ConfigDict(models=[dict(abbr='model')],
                                              datasets=[],
                                              work_dir=str(tmp_path)),
                                   summary_groups=[
                                       dict(name='combined',
                                            subsets=['a', 'b'],
                                            metric='Bayes@2 posterior SD')
                                   ])
    with pytest.raises(ValueError, match='cross-dataset'):
        summarizer._calculate_group_metrics({}, {}, {}, {})
