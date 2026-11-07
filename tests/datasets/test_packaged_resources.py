import pytest

from opencompass.datasets.leval.evaluators import _load_battle_samples
from opencompass.datasets.medbench.medbench import process_generated_results_CDN


def test_medbench_cdn_postprocess_is_cwd_independent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = process_generated_results_CDN(['高血压性心脏病'])

    assert result == [[{
        'entity': '高血压性心脏病',
        'type': 'normalization',
    }]]


@pytest.mark.parametrize('battle_model', [
    'claude-100k',
    'turbo-16k-0613',
])
def test_leval_battle_samples_are_cwd_independent(battle_model, tmp_path,
                                                   monkeypatch):
    monkeypatch.chdir(tmp_path)

    samples = _load_battle_samples(battle_model)

    assert samples
    assert all({'query', 'gt', f'{battle_model}_pred'} <= sample.keys()
               for sample in samples)
