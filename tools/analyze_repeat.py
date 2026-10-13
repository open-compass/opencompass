import argparse
import json
from collections import defaultdict
from pathlib import Path

from opencompass.utils.repeat_analysis import (  # isort: skip
    GZIP_THRESHOLD, MIN_PATTERN_COUNT, NUM_SAMPLE_PATTERNS, PATTERN_SIZE,
    TokenizerWrapper, _analyze_benchmark, _build_abnormal_samples,
    _fill_abnormal_predictions, _iter_prediction_items, _prediction_to_text,
    print_analysis_summary)

prog_description = """\
Analyze prediction repeat.
"""


def parse_args():
    parser = argparse.ArgumentParser(description=prog_description)
    parser.add_argument('work_dir', type=Path, help='The root work_dir')
    parser.add_argument('--model', type=str, help='The model abbr to analyze')
    parser.add_argument('--tokenizer',
                        type=str,
                        default='gpt-4o',
                        help='HF tokenizer path or tiktoken encoder name.')
    parser.add_argument('--batch-size',
                        type=int,
                        default=1024,
                        help='Tokenize batch size.')
    parser.add_argument('--think-tag',
                        type=str,
                        help='Think tag to split reasoning and content.')
    parser.add_argument('--out', type=Path, help='output file path')
    parser.add_argument('--no-progress',
                        action='store_true',
                        help='Disable benchmark progress bars.')
    args = parser.parse_args()
    return args


def build_tokenizer(name: str) -> TokenizerWrapper:
    try:
        import tiktoken
        if name in tiktoken.model.MODEL_TO_ENCODING:
            tok = tiktoken.encoding_for_model(name)
            wrapper = TokenizerWrapper(tok, 'tiktoken')
            return wrapper
    except Exception:
        pass

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name,
                                            trust_remote_code=True,
                                            use_fast=True)
        wrapper = TokenizerWrapper(tok, 'hf', name)
        return wrapper
    except Exception:
        pass

    try:
        import tiktoken
        tok = tiktoken.encoding_for_model('gpt-4')
        wrapper = TokenizerWrapper(tok, 'tiktoken')
        return wrapper
    except Exception:
        pass

    raise RuntimeError(f'Cannot initialize tokenizer {name}')


def collect_predictions_by_benchmark(
    work_dir: Path,
    model_abbr: str | None = None,
) -> dict[str, list[dict]]:
    """Read all prediction files, grouped by benchmark.

    Each record has: model, benchmark, sample_id, prediction_path, text,
    and optionally res_length.

    Returns:
        grouped_records
    """
    grouped = defaultdict(list)
    prediction_root = Path(work_dir) / 'predictions'

    if model_abbr is None:
        model_abbr = next(prediction_root.iterdir()).name
        print(f'Automatically select {model_abbr} to analyze.')

    model_preds = prediction_root / model_abbr
    assert model_preds.exists(), f'{model_preds} does not exist.'

    for pred_file in model_preds.glob('*.json'):
        if pred_file.stem.rpartition('_')[-1].isdigit():
            dataset_abbr = pred_file.stem.rpartition('_')[0]
        else:
            dataset_abbr = pred_file.stem

        for sample_id, sample in _iter_prediction_items(pred_file):
            res_length = None
            if isinstance(sample, dict):
                prediction = sample.get('prediction', '')
                raw_res_length = sample.get('res_length', None)
                if raw_res_length is not None:
                    if isinstance(raw_res_length, list):
                        res_length = (raw_res_length[0]
                                      if raw_res_length else 0)
                    else:
                        res_length = raw_res_length
            else:
                prediction = sample
            text = _prediction_to_text(prediction)

            grouped[dataset_abbr].append({
                'model': model_abbr,
                'benchmark': dataset_abbr,
                'sample_id': sample_id,
                'prediction_path': str(pred_file),
                'text': text,
                'res_length': res_length,
            })
    if len(grouped) == 0:
        raise RuntimeError(f'No available prediction files in {model_preds}.')
    return grouped


def main():
    args = parse_args()

    # Build tokenizers
    tokenizer = build_tokenizer(args.tokenizer)

    # Read all predictions grouped by benchmark
    grouped_records = collect_predictions_by_benchmark(args.work_dir,
                                                       args.model)
    model_abbr = next(iter(grouped_records.values()))[0]['model']

    # Analyze each benchmark independently
    all_records: list[dict] = []
    benchmark_stats: dict = {}

    for benchmark, records in grouped_records.items():
        print(f'\n[{benchmark}] ({len(records)} samples)')

        stats = _analyze_benchmark(records, {model_abbr: tokenizer},
                                   show_progress=not args.no_progress,
                                   batch_size=args.batch_size,
                                   think_tag=args.think_tag)
        benchmark_stats[benchmark] = stats
        all_records.extend(records)
    tokenizer.close()

    # Build abnormal samples and fill predictions
    abnormal_samples = _build_abnormal_samples(all_records,
                                               benchmark_stats,
                                               with_reasoning=bool(
                                                   args.think_tag))
    _fill_abnormal_predictions(abnormal_samples)

    report = {
        'work_dir': str(args.work_dir),
        'settings': {
            'pattern_size':
            PATTERN_SIZE,
            'num_sample_patterns':
            NUM_SAMPLE_PATTERNS,
            'length_unit':
            'tokens',
            'repeat_analysis_threshold':
            'p75 of benchmark token lengths',
            'pattern_rule':
            f'periodic pattern with count > {MIN_PATTERN_COUNT}',
            'gzip_rule':
            f'gzip_compression_ratio > {GZIP_THRESHOLD}',
            'repeat_analysis_min_length_rule':
            'pattern/gzip skipped when length < p75',
        },
        'summary': {
            'model':
            model_abbr,
            'num_benchmarks':
            len(benchmark_stats),
            'num_samples':
            len(all_records),
            'num_repeat_analysis_skipped':
            sum(r['repeat_analysis_skipped'] for r in all_records),
            'abnormal_counts':
            {key: len(value)
             for key, value in abnormal_samples.items()},
        },
        'benchmark_stats': benchmark_stats,
        'abnormal_samples': abnormal_samples,
    }

    if args.out is None:
        output_path = Path(
            args.work_dir) / 'summary' / (f'repeat_analysis_{model_abbr}.json')
    else:
        output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=4, ensure_ascii=False),
                           encoding='utf-8')
    print_analysis_summary(report)
    return str(output_path)


if __name__ == '__main__':
    main()
