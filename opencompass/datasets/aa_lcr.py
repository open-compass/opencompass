import csv
import os
import posixpath
import re
import unicodedata
import zipfile
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from datasets import Dataset, DatasetDict

from opencompass.registry import DICT_POSTPROCESSORS, LOAD_DATASET

from .base import BaseDataset

DEFAULT_REPO_ID = 'ArtificialAnalysis/AA-LCR'
DEFAULT_CSV_FILENAME = 'AA-LCR_Dataset.csv'
DEFAULT_ZIP_FILENAME = 'extracted_text/AA-LCR_extracted-text.zip'

PROMPT_TEMPLATE = """BEGIN INPUT DOCUMENTS

{documents_text}

END INPUT DOCUMENTS

Answer the following question using the input documents provided above.

START QUESTION

{question}

END QUESTION
"""


def _split_semicolon_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(';') if item.strip()]


def _resolve_local_file(root: str, filename: str) -> Optional[str]:
    candidates = [
        os.path.join(root, filename),
        os.path.join(root, os.path.basename(filename)),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def _resolve_data_files(path: str, csv_filename: str,
                        zip_filename: str) -> Tuple[str, str]:
    if os.path.isdir(path):
        csv_path = _resolve_local_file(path, csv_filename)
        zip_path = _resolve_local_file(path, zip_filename)
        if csv_path and zip_path:
            return csv_path, zip_path

    if os.path.isfile(path):
        csv_path = path
        zip_path = _resolve_local_file(os.path.dirname(path), zip_filename)
        if zip_path:
            return csv_path, zip_path

    from huggingface_hub import hf_hub_download

    csv_path = hf_hub_download(repo_id=path,
                               filename=csv_filename,
                               repo_type='dataset')
    zip_path = hf_hub_download(repo_id=path,
                               filename=zip_filename,
                               repo_type='dataset')
    return csv_path, zip_path


def _doc_name(category: str, set_id: str, filename: str) -> str:
    return posixpath.join('lcr', category, set_id, filename)


def _recover_zip_name(filename: str) -> str:
    try:
        return filename.encode('cp437').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return filename


def _normalize_doc_name(filename: str) -> str:
    return unicodedata.normalize('NFC', _recover_zip_name(filename))


def _read_doc_from_zip(zip_file: zipfile.ZipFile, category: str, set_id: str,
                       filename: str) -> str:
    doc_name = _doc_name(category, set_id, filename)
    if doc_name in zip_file.namelist():
        with zip_file.open(doc_name) as f:
            return f.read().decode('utf-8')

    recovered_names = {
        _normalize_doc_name(zip_name): zip_name
        for zip_name in zip_file.namelist()
    }
    zip_name = recovered_names.get(_normalize_doc_name(doc_name))
    if zip_name is not None:
        with zip_file.open(zip_name) as f:
            return f.read().decode('utf-8')

    raise FileNotFoundError(f'Cannot find AA-LCR document {filename!r} for '
                            f'{category}/{set_id} in {zip_file.filename}.')


def _build_prompt(documents: List[str], question: str) -> str:
    documents_text = '\n\n'.join(
        'BEGIN DOCUMENT {}:\n{}\nEND DOCUMENT {}'.format(
            idx + 1, doc, idx + 1) for idx, doc in enumerate(documents))
    return PROMPT_TEMPLATE.format(documents_text=documents_text,
                                  question=question)


@LOAD_DATASET.register_module()
class AALCRDataset(BaseDataset):

    @staticmethod
    def load(path: str = DEFAULT_REPO_ID,
             csv_filename: str = DEFAULT_CSV_FILENAME,
             zip_filename: str = DEFAULT_ZIP_FILENAME):
        csv_path, zip_path = _resolve_data_files(path, csv_filename,
                                                 zip_filename)

        raw_data = []
        with open(csv_path, encoding='utf-8', newline='') as f, \
                zipfile.ZipFile(zip_path) as zip_file:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader):
                data_source_filenames = _split_semicolon_list(
                    row['data_source_filenames'])
                data_source_urls = _split_semicolon_list(
                    row.get('data_source_urls', ''))
                category = row['document_category']
                set_id = row['document_set_id']
                documents = [
                    _read_doc_from_zip(zip_file, category, set_id, filename)
                    for filename in data_source_filenames
                ]
                question = row['question']
                answer = _split_semicolon_list(row['answer'])

                raw_data.append({
                    'id': int(row.get('', row_idx)),
                    'question_id': int(row['question_id']),
                    'document_category': category,
                    'document_set_id': set_id,
                    'question': question,
                    'answer': answer,
                    'data_source_filenames': data_source_filenames,
                    'data_source_urls': data_source_urls,
                    'input_tokens': int(row['input_tokens']),
                    'prompt': _build_prompt(documents, question),
                })

        dataset = Dataset.from_list(raw_data)
        return DatasetDict({'test': dataset, 'train': dataset})


def _parse_aa_lcr_judgement(judgement: str) -> str:
    normalized = judgement.strip().upper()
    if re.search(r'\bINCORRECT\b', normalized):
        return 'INCORRECT'
    if re.search(r'\bCORRECT\b', normalized):
        return 'CORRECT'
    return 'UNKNOWN'


def _get_test_samples(dataset) -> List[Dict]:
    if dataset is None:
        return []
    try:
        return list(dataset.reader.dataset['test'])
    except AttributeError:
        return []


@DICT_POSTPROCESSORS.register_module()
def aa_lcr_llmjudge_postprocess(output: dict,
                                output_path: str,
                                dataset=None) -> dict:
    samples = _get_test_samples(dataset)
    details = []
    category_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    correct = 0
    incorrect = 0
    unknown = 0

    for key, value in sorted(output.items(), key=lambda item: int(item[0])):
        idx = int(key)
        sample = samples[idx] if idx < len(samples) else {}
        judge_response = value.get('prediction', '')
        judgement = _parse_aa_lcr_judgement(judge_response)

        if judgement == 'CORRECT':
            correct += 1
        elif judgement == 'INCORRECT':
            incorrect += 1
        else:
            unknown += 1

        category = sample.get('document_category', 'unknown')
        category_stats[category]['total'] += 1
        if judgement == 'CORRECT':
            category_stats[category]['correct'] += 1

        details.append({
            'id': idx,
            'question_id': sample.get('question_id'),
            'document_category': category,
            'document_set_id': sample.get('document_set_id'),
            'question': sample.get('question'),
            'gold': sample.get('answer'),
            'candidate_answer': sample.get('prediction'),
            'judge_response': judge_response,
            'judgement': judgement,
            'correct': judgement == 'CORRECT',
        })

    total = len(details)
    results = {
        'accuracy': correct / total * 100 if total else 0,
        'correct_count': correct,
        'incorrect_count': incorrect,
        'unknown_count': unknown,
        'total': total,
        'details': details,
    }
    for category, stats in category_stats.items():
        safe_category = re.sub(r'\W+', '_', category).strip('_')
        results[f'accuracy_{safe_category}'] = (stats['correct'] /
                                                stats['total'] *
                                                100 if stats['total'] else 0)

    return results
