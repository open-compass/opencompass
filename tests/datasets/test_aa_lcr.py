import csv
import zipfile

from datasets import Dataset

from opencompass.configs.datasets.aa_lcr.aa_lcr_gen import \
    AA_LCR_JUDGE_TEMPLATE
from opencompass.datasets.aa_lcr import (AALCRDataset,
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
                'prediction': 'CORRECT'
            },
            '1': {
                'prediction': 'INCORRECT'
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
