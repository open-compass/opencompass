import json

from opencompass.datasets.c4_bench import (C4BenchDataset,
                                           C4BenchEvaluator,
                                           parse_c4_answer)


def _row(task, instance_id):
    return {
        'instance_id': instance_id,
        'task': task,
        'question': '请回答成语。',
        'answer': '一叶障目',
        'answer_aliases': [],
        'image': 'https://example.com/image.png',
    }


def test_c4_loader_filters_primary_tasks_and_builds_multimodal_chatml(
        tmp_path):
    data_path = tmp_path / 'eval.jsonl'
    rows = [_row('H0', 'item__H0'), _row('E1', 'item__E1')]
    data_path.write_text(
        '\n'.join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding='utf-8',
    )

    dataset = C4BenchDataset.load(path=str(data_path), split='primary')['test']

    assert len(dataset) == 1
    content = dataset[0]['chatml_question'][0]['content']
    assert content[0]['type'] == 'image'
    assert content[0]['image_url'] == 'https://example.com/image.png'
    assert content[1]['text'] == '请回答成语。'


def test_c4_evaluator_uses_official_primary_denominator():
    evaluator = C4BenchEvaluator()
    references = [
        {
            'task': 'H0',
            'answer': '一叶障目',
            'answer_aliases': [],
        },
        {
            'task': 'E0',
            'answer': '一叶障目',
            'answer_aliases': [],
        },
        {
            'task': 'E1',
            'answer': '一叶障目',
            'answer_aliases': [],
        },
    ]
    predictions = [
        '一叶障目',
        '{"answer": "错误答案", "explanation": "..."}',
        '{"answer": "一叶障目", "explanation": "..."}',
    ]

    scores = evaluator.score(predictions, references)

    assert scores['Primary Score'] == 50
    assert scores['E0 JSON Valid'] == 100
    assert scores['E1 JSON Valid'] == 100
    assert parse_c4_answer('E0', '最终答案：一叶障目') == ('一叶障目', False)
