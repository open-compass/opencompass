import json
from unittest.mock import patch

import mmengine
import pytest

from opencompass.datasets.hardmtbench import (HardMTBenchDataset,
                                              HardMTBenchEvaluator,
                                              parse_gemba_da, term_accuracy)
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate

TERMS = [
    {
        'source': '隐性债务',
        'target': 'implicit debt',
        'category': '金融术语'
    },
    {
        'source': '去杠杆',
        'target': 'deleveraging',
        'category': '金融术语'
    },
]


def _item(pair, direction, domain, hardness):
    zh, en = f'中文{pair}', f'English {pair}'
    zh2en = direction == 'zh2en'
    return {
        'source_text': zh if zh2en else en,
        'reference': en if zh2en else zh,
        'id': f'{pair}_{direction}',
        'source_language': 'Chinese' if zh2en else 'English',
        'target_language': 'English' if zh2en else 'Chinese',
        'domain': domain,
        'sub_domain': 'x',
        'domain_keywords': [],
        'terminology': TERMS,
        'scores': {
            'hardness': hardness
        },
    }


@pytest.fixture
def data_file(tmp_path):
    items = []
    for pair, domain, hardness in [('p1', 'Finance', 65.0),
                                   ('p2', 'Technology', 85.0)]:
        for direction in ('zh2en', 'en2zh'):
            items.append(_item(pair, direction, domain, hardness))
    path = tmp_path / 'HardMTBench.jsonl'
    path.write_text('\n'.join(
        json.dumps(x, ensure_ascii=False) for x in items),
                    encoding='utf-8')
    return str(path)


def test_loader_filters_direction_and_domain(data_file):
    zh2en = HardMTBenchDataset.load(path=data_file, direction='zh2en')
    en2zh = HardMTBenchDataset.load(path=data_file, direction='en2zh')
    tech = HardMTBenchDataset.load(path=data_file,
                                   direction='zh2en',
                                   domain='Technology')

    assert zh2en['id'] == ['p1_zh2en', 'p2_zh2en']
    assert zh2en[0]['target_language_zh'] == '英语'
    assert en2zh[0]['target_language_zh'] == '中文'
    assert en2zh[0]['source_text'] == 'English p1'
    assert tech['id'] == ['p2_zh2en']
    with pytest.raises(AssertionError):
        HardMTBenchDataset.load(path=data_file,
                                direction='zh2en',
                                domain='Tech')


@pytest.mark.parametrize('answer, expected', [
    ('85', 85),
    ('Score: 92', 92),
    ("['70']", 70),
    ('100', 100),
    ('101', None),
    ('80 or 90', None),
    ('no idea', None),
    (None, None),
])
def test_parse_gemba_da(answer, expected):
    assert parse_gemba_da(answer) == expected


def test_term_accuracy_uses_target_side_per_direction():
    assert term_accuracy('the Implicit  Debt grew', TERMS,
                         'English') == pytest.approx(0.5)
    assert term_accuracy('隐性债务和去杠杆', TERMS, 'Chinese') == 1.0
    # full-width forms are folded by NFKC
    assert term_accuracy('ｄｅｌｅｖｅｒａｇｉｎｇ', TERMS, 'English') == 0.5
    assert term_accuracy('x', [], 'English') is None


class _FakeInferencer:
    """Return a judge answer per (temperature, prediction)."""

    answers = {}
    temperatures = []

    def __init__(self, model, output_json_filepath, output_json_filename,
                 **kwargs):
        self.temperature = model
        self.path = f'{output_json_filepath}/{output_json_filename}'

    def inference(self, retriever, prompt_template):
        self.temperatures.append(self.temperature)
        out = {}
        for i in range(len(retriever.test_ds)):
            prompt = retriever.generate_prompt_for_generate_task(
                i, '', prompt_template=prompt_template)[0]['content']
            out[str(i)] = {
                'prediction': self.answers[(self.temperature, prompt)]
            }
        mmengine.dump(out, self.path)


def _evaluator(tmp_path, metrics):
    evaluator = HardMTBenchEvaluator(
        prompt_template=dict(type=RawPromptTemplate,
                             messages=[{
                                 'role': 'user',
                                 'content': '{prediction}'
                             }]),
        judge_cfg=dict(type='OpenAISDK', path='judge', temperature=0.0),
        metrics=metrics,
    )
    evaluator._out_dir = str(tmp_path / 'results' / 'hardmtbench_zh2en')
    (tmp_path / 'results').mkdir(exist_ok=True)
    return evaluator


@patch('opencompass.datasets.hardmtbench.build_model_from_cfg',
       side_effect=lambda model_cfg: model_cfg['temperature'])
@patch('opencompass.datasets.hardmtbench.GenInferencer', _FakeInferencer)
def test_evaluator_runs_metrics_serially(_, data_file, tmp_path):
    test_set = HardMTBenchDataset.load(path=data_file, direction='zh2en')
    preds = ['implicit debt and deleveraging', 'about debt']
    _FakeInferencer.temperatures = []
    _FakeInferencer.answers = {
        (0.0, preds[0]):
        '90',
        # invalid at temperature 0, retried at 0.1 like the official GptApi
        (0.0, preds[1]):
        'I think 60 or 70',
        (0.1, preds[1]):
        '60',
    }
    evaluator = _evaluator(tmp_path, ['gemba_da', 'xcomet_xxl', 'term_acc'])
    with patch.object(HardMTBenchEvaluator,
                      '_xcomet_xxl',
                      return_value=[0.8, 0.5]) as xcomet:
        results = evaluator.score(preds, test_set['reference'], test_set)

    assert _FakeInferencer.temperatures == [0.0, 0.1]
    xcomet.assert_called_once()
    assert results['gemba_da'] == pytest.approx(75)
    assert results['gemba_parse_failures'] == 0
    assert results['xcomet_xxl'] == pytest.approx(65)
    assert results['term_acc'] == pytest.approx(50)
    assert results['Finance/gemba_da'] == pytest.approx(90)
    assert results['Technology/term_acc'] == pytest.approx(0)
    assert results['hardness_50-70/xcomet_xxl'] == pytest.approx(80)
    assert results['hardness_80-90/gemba_da'] == pytest.approx(60)
    assert 'hardness_90-100/gemba_da' not in results
    assert results['details'][1]['gemba_response'] == '60'
    assert (tmp_path / 'results' /
            'hardmtbench_zh2en_replica0_gemba_retry1.json').exists()


@patch('opencompass.datasets.hardmtbench.GenInferencer')
@patch('opencompass.datasets.hardmtbench.build_model_from_cfg')
def test_evaluator_reuses_outputs_and_drops_invalid(mock_build, mock_infer,
                                                    data_file, tmp_path):
    test_set = HardMTBenchDataset.load(path=data_file,
                                       direction='zh2en',
                                       domain='Technology')
    evaluator = _evaluator(tmp_path, ['gemba_da', 'term_acc'])
    evaluator.gemba_max_retries = 1
    for name, answer in [('gemba', 'n/a'), ('gemba_retry1', 'still n/a')]:
        mmengine.dump({'0': {
            'prediction': answer
        }}, tmp_path / 'results' / f'hardmtbench_zh2en_replica0_{name}.json')

    results = evaluator.score(['deleveraging'], test_set['reference'],
                              test_set)

    mock_build.assert_not_called()
    mock_infer.assert_not_called()
    # never-valid answers are left out of the mean like the official Scores
    assert results['gemba_parse_failures'] == 1
    assert results['gemba_da'] != results['gemba_da']  # NaN
    assert results['term_acc'] == pytest.approx(50)
    # single domain: no per-domain breakdown
    assert 'Technology/term_acc' not in results
    assert 'xcomet_xxl' not in results


class _FakeResponse:

    def __init__(self, scores):
        self.scores = scores

    def raise_for_status(self):
        pass

    def json(self):
        return {'scores': self.scores}


def test_xcomet_calls_service_in_chunks_with_retry(data_file, tmp_path,
                                                   monkeypatch):
    test_set = HardMTBenchDataset.load(path=data_file, direction='en2zh')
    evaluator = _evaluator(tmp_path, ['xcomet_xxl'])
    evaluator.xcomet_request_size = 1
    monkeypatch.setenv('OC_XCOMET_API_BASE', 'http://xcomet:8090/')
    monkeypatch.setattr('time.sleep', lambda s: None)
    calls = []

    def fake_post(url, json, timeout):
        calls.append((url, json['samples']))
        if len(calls) == 1:
            raise ConnectionError('service restarting')
        return _FakeResponse(
            [0.9 if 'p1' in json['samples'][0]['ref'] else 0.3])

    monkeypatch.setattr('requests.post', fake_post)
    results = evaluator.score(['译文一', '译文二'], test_set['reference'], test_set)

    # first request failed and was retried, then one request per sample
    assert [url for url, _ in calls] == ['http://xcomet:8090/score'] * 3
    assert calls[1][1] == [{'src': 'English p1', 'mt': '译文一', 'ref': '中文p1'}]
    assert results['xcomet_xxl'] == pytest.approx(60)
    assert results['details'][1]['xcomet_xxl'] == pytest.approx(30)

    # scores are cached, the service is not called again
    monkeypatch.setattr('requests.post', None)
    assert evaluator.score(['译文一', '译文二'], test_set['reference'],
                           test_set)['xcomet_xxl'] == pytest.approx(60)


def test_xcomet_requires_service_address(data_file, tmp_path, monkeypatch):
    test_set = HardMTBenchDataset.load(path=data_file, direction='zh2en')
    monkeypatch.delenv('OC_XCOMET_API_BASE', raising=False)
    evaluator = _evaluator(tmp_path, ['xcomet_xxl'])
    with pytest.raises(AssertionError, match='OC_XCOMET_API_BASE'):
        evaluator.score(['a', 'b'], test_set['reference'], test_set)
