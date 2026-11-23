import mmengine
from mmengine.config import ConfigDict

from opencompass.tasks.subjective_eval import SubjectiveEvalTask


def _make_task(work_dir):
    task = SubjectiveEvalTask.__new__(SubjectiveEvalTask)
    task.work_dir = str(work_dir)
    task.logger = mmengine.MMLogger.get_instance('test_subjective_eval')
    return task


def test_load_sampled_prediction_keeps_dataset_abbr(tmp_path):
    prediction_dir = tmp_path / 'predictions' / 'mock_test'
    prediction_dir.mkdir(parents=True)
    mmengine.dump({str(i): {
        'prediction': f'answer-{i}'
    }
                   for i in range(12)},
                  prediction_dir / 'alignment_bench_v1_1.json')

    task = _make_task(tmp_path)
    result = task._load_model_pred(
        ConfigDict(abbr='mock_test'),
        ConfigDict(
            abbr='alignment_bench_v1_1',
            reader_cfg=ConfigDict(test_range='[0:10]'),
        ), ConfigDict(evaluator=ConfigDict()), [])

    assert result['model_preds'] == [f'answer-{i}' for i in range(10)]


def test_load_sampled_given_prediction_keeps_dataset_abbr(tmp_path):
    given_dir = tmp_path / 'given'
    given_dir.mkdir()
    mmengine.dump({str(i): {
        'prediction': f'given-{i}'
    }
                   for i in range(12)}, given_dir / 'arena_hard.json')

    task = _make_task(tmp_path)
    result = task._load_model_pred(
        ConfigDict(abbr='gpt4'),
        ConfigDict(
            abbr='arena_hard',
            reader_cfg=ConfigDict(test_range='[0:10]'),
        ), ConfigDict(evaluator=ConfigDict()), [{
            'abbr': 'gpt4',
            'path': str(given_dir),
        }])

    assert result['model_preds'] == [f'given-{i}' for i in range(10)]
