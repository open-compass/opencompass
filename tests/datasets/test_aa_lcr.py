import csv
import zipfile
from unittest.mock import call, patch

import pytest
from datasets import Dataset
from mmengine.config import Config

from opencompass.configs.datasets.aa_lcr.aa_lcr_gen import \
    AA_LCR_JUDGE_TEMPLATE
from opencompass.datasets.aa_lcr import (AALCRDataset, _parse_aa_lcr_judgement,
                                         _resolve_data_files,
                                         _resolve_data_path,
                                         aa_lcr_llmjudge_postprocess)
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate


def test_aa_lcr_local_loader_preserves_document_order(tmp_path):
    csv_path = tmp_path / 'AA-LCR_Dataset.csv'
    zip_dir = tmp_path / 'extracted_text'
    zip_dir.mkdir()
    zip_path = zip_dir / 'AA-LCR_extracted-text.zip'

    with csv_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                '',
                'document_category',
                'document_set_id',
                'question_id',
                'question',
                'answer',
                'data_source_filenames',
                'data_source_urls',
                'input_tokens',
            ])
        writer.writeheader()
        writer.writerow({
            '': '0',
            'document_category': 'Test_Category',
            'document_set_id': 'test_set',
            'question_id': '1',
            'question': 'What is the answer?',
            'answer': 'Second document wins.',
            'data_source_filenames': 'first.txt;second.txt',
            'data_source_urls': 'https://example.com/1;https://example.com/2',
            'input_tokens': '42',
        })

    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('lcr/Test_Category/test_set/first.txt', 'first text')
        zf.writestr('lcr/Test_Category/test_set/second.txt', 'second text')

    dataset = AALCRDataset.load(path=str(tmp_path))['test']

    assert len(dataset) == 1
    sample = dataset[0]
    assert sample['answer'] == ['Second document wins.']
    assert sample['data_source_filenames'] == ['first.txt', 'second.txt']
    assert sample['prompt'].index('BEGIN DOCUMENT 1:\nfirst text') < (
        sample['prompt'].index('BEGIN DOCUMENT 2:\nsecond text'))
    assert sample['prompt'].strip().endswith('END QUESTION')


def test_aa_lcr_local_loader_recovers_zip_filename_mojibake(tmp_path):
    csv_path = tmp_path / 'AA-LCR_Dataset.csv'
    zip_dir = tmp_path / 'extracted_text'
    zip_dir.mkdir()
    zip_path = zip_dir / 'AA-LCR_extracted-text.zip'

    with csv_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                '',
                'document_category',
                'document_set_id',
                'question_id',
                'question',
                'answer',
                'data_source_filenames',
                'data_source_urls',
                'input_tokens',
            ])
        writer.writeheader()
        writer.writerow({
            '': '0',
            'document_category': 'Test_Category',
            'document_set_id': 'test_set',
            'question_id': '1',
            'question': 'What is the answer?',
            'answer': 'Recovered document.',
            'data_source_filenames': 'No cheat-codes needed – guide.txt',
            'data_source_urls': '',
            'input_tokens': '42',
        })

    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr(
            'lcr/Test_Category/test_set/'
            'No cheat-codes needed ΓÇô guide.txt',
            'recovered text',
        )

    sample = AALCRDataset.load(path=str(tmp_path))['test'][0]

    assert 'BEGIN DOCUMENT 1:\nrecovered text' in sample['prompt']


def test_aa_lcr_data_path_falls_back_when_cache_resolution_fails():
    with patch('opencompass.datasets.aa_lcr.get_data_path',
               side_effect=AssertionError):
        path = _resolve_data_path(
            'opencompass/AA-LCR',
            'AA-LCR_Dataset.csv',
            'extracted_text/AA-LCR_extracted-text.zip',
        )

    assert path == 'ArtificialAnalysis/AA-LCR'


def test_aa_lcr_data_path_falls_back_when_local_data_is_incomplete(tmp_path):
    with patch('opencompass.datasets.aa_lcr.get_data_path',
               return_value=str(tmp_path)):
        path = _resolve_data_path(
            'opencompass/AA-LCR',
            'AA-LCR_Dataset.csv',
            'extracted_text/AA-LCR_extracted-text.zip',
        )

    assert path == 'ArtificialAnalysis/AA-LCR'


def test_aa_lcr_data_path_falls_back_when_resolved_path_does_not_exist(
        tmp_path):
    missing_path = tmp_path / 'missing'
    with patch('opencompass.datasets.aa_lcr.get_data_path',
               return_value=str(missing_path)):
        path = _resolve_data_path(
            'opencompass/AA-LCR',
            'AA-LCR_Dataset.csv',
            'extracted_text/AA-LCR_extracted-text.zip',
        )

    assert path == 'ArtificialAnalysis/AA-LCR'


def test_aa_lcr_explicit_hub_repo_bypasses_unversioned_cache(tmp_path):
    with patch('opencompass.datasets.aa_lcr.get_data_path',
               return_value=str(tmp_path)) as get_data_path:
        path = _resolve_data_path(
            'ArtificialAnalysis/AA-LCR',
            'AA-LCR_Dataset.csv',
            'extracted_text/AA-LCR_extracted-text.zip',
        )

    assert path == 'ArtificialAnalysis/AA-LCR'
    get_data_path.assert_not_called()


@pytest.mark.parametrize('revision', [
    'bdae010bbce259820c0e34c1d7cce210d966fb75',
    '9a77ef56b717057ade24ceab4d273712a0b4f19e',
])
def test_aa_lcr_hub_files_use_the_same_revision(revision):
    with patch('huggingface_hub.hf_hub_download',
               side_effect=['/tmp/questions.csv', '/tmp/documents.zip']) as dl:
        files = _resolve_data_files(
            'ArtificialAnalysis/AA-LCR',
            'AA-LCR_Dataset.csv',
            'extracted_text/AA-LCR_extracted-text.zip',
            revision=revision,
        )

    assert files == ('/tmp/questions.csv', '/tmp/documents.zip')
    assert dl.call_args_list == [
        call(repo_id='ArtificialAnalysis/AA-LCR',
             filename='AA-LCR_Dataset.csv',
             repo_type='dataset',
             revision=revision),
        call(repo_id='ArtificialAnalysis/AA-LCR',
             filename='extracted_text/AA-LCR_extracted-text.zip',
             repo_type='dataset',
             revision=revision),
    ]


@pytest.mark.parametrize(('response', 'expected'), [
    ('CORRECT', 'CORRECT'),
    ('INCORRECT', 'INCORRECT'),
    ('{"verdict": "CORRECT"}', 'CORRECT'),
    ('{"verdict": "INCORRECT"}', 'INCORRECT'),
    ('```json\n{"verdict": "correct"}\n```', 'CORRECT'),
    ('{"reason": "Not INCORRECT", "verdict": "CORRECT"}', 'CORRECT'),
    ('{"reason": "Some steps are CORRECT", "verdict": "INCORRECT"}',
     'INCORRECT'),
    ('{"reason": "CORRECT", "verdict": "UNKNOWN"}', 'UNKNOWN'),
    ('{"reason": "CORRECT"}', 'UNKNOWN'),
    ('No verdict', 'UNKNOWN'),
])
def test_aa_lcr_judgement_parses_json_verdict_before_explanation(
        response, expected):
    assert _parse_aa_lcr_judgement(response) == expected


def test_aa_lcr_v1_1_config_keeps_inference_and_updates_judging():
    old_cfg = Config.fromfile(
        'opencompass/configs/datasets/aa_lcr/aa_lcr_gen.py')
    new_cfg = Config.fromfile(
        'opencompass/configs/datasets/aa_lcr/aa_lcr_v1_1_gen.py')
    old = old_cfg.aa_lcr_datasets[0]
    new = new_cfg.aa_lcr_v1_1_datasets[0]
    assert old.revision == 'bdae010bbce259820c0e34c1d7cce210d966fb75'
    assert old.eval_cfg.evaluator.dataset_cfg.revision == old.revision
    assert new.revision == '9a77ef56b717057ade24ceab4d273712a0b4f19e'
    assert new.eval_cfg.evaluator.dataset_cfg.revision == new.revision
    assert new.path == 'opencompass/AA-LCR-v1.1'
    assert new.eval_cfg.evaluator.dataset_cfg.path == new.path
    assert new.abbr != old.abbr
    assert new.type is old.type is AALCRDataset
    assert new.n == old.n == 3
    assert new.reader_cfg == old.reader_cfg
    assert new.infer_cfg == old.infer_cfg
    assert new.eval_cfg.evaluator.type is old.eval_cfg.evaluator.type
    assert new.eval_cfg.evaluator.dict_postprocessor == (
        old.eval_cfg.evaluator.dict_postprocessor)

    template = RawPromptTemplate(
        messages=new.eval_cfg.evaluator.prompt_template.messages)
    messages = template.generate_item({
        'question': 'What percentage?',
        'answer': ['65.8%'],
        'prediction': '65.8%',
    })
    assert [message['role'] for message in messages] == ['system', 'user']
    assert '0.675, "67.5%"' in messages[0]['content']
    assert 'it must commit to one' in messages[0]['content']
    assert 'START QUESTION What percentage?' in messages[1]['content']
    assert "The OFFICIAL ANSWER: ['65.8%']" in messages[1]['content']
    assert '\n65.8%\n' in messages[1]['content']
    assert 'Reply as JSON, with a verdict' in messages[1]['content']


def test_aa_lcr_llmjudge_postprocess():
    test_set = Dataset.from_list([
        {
            'question_id': 1,
            'document_category': 'Category A',
            'document_set_id': 'set_a',
            'question': 'Question A',
            'answer': 'Answer A',
            'prediction': 'Answer A',
        },
        {
            'question_id': 2,
            'document_category': 'Category A',
            'document_set_id': 'set_a',
            'question': 'Question B',
            'answer': 'Answer B',
            'prediction': 'Wrong',
        },
    ])

    class _Reader:
        pass

    _Reader.dataset = {'test': test_set}

    class _JudgeDataset:
        reader = _Reader()

    scores = aa_lcr_llmjudge_postprocess(
        {
            '0': {
                'prediction': '{"reason": "Not INCORRECT", "verdict": "CORRECT"}'
            },
            '1': {
                'prediction': '{"verdict": "INCORRECT"}'
            },
        },
        output_path='unused.json',
        dataset=_JudgeDataset(),
    )

    assert scores['accuracy'] == 50
    assert scores['correct_count'] == 1
    assert scores['incorrect_count'] == 1
    assert scores['accuracy_Category_A'] == 50


def test_aa_lcr_judge_template_uses_opencompass_fields():
    template = RawPromptTemplate(
        messages=[dict(role='user', content=AA_LCR_JUDGE_TEMPLATE)])

    messages = template.generate_item({
        'question': 'Which document wins?',
        'answer': ['Second document wins.'],
        'prediction': 'The second document wins.',
    })
    content = messages[0]['content']

    assert '{official_answer}' not in content
    assert '{candidate_answer}' not in content
    assert 'Second document wins.' in content
    assert 'The second document wins.' in content
