# flake8: noqa: W605
import json
import math
import os.path as osp
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.linear_model import LogisticRegression

from opencompass.datasets.subjective.compass_arena_subjective_bench import \
    get_element_counts
from opencompass.registry import DICT_POSTPROCESSORS, LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset
from .utils import get_judgeanswer_and_reference


@LOAD_DATASET.register_module()
class ArenaHardDataset(BaseDataset):

    def load(self, path: str, name: str, *args, **kwargs):
        path = get_data_path(path, local_mode=True)
        filename = osp.join(path, f'{name}.jsonl')
        dataset = DatasetDict()
        raw_data = []
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                problem = json.loads(line)
                question_id = problem['question_id']
                cluster = problem['cluster']
                question = problem['turns'][0][
                    'content']  # only one turn in arena_hard
                raw_data.append({
                    'question': question,
                    'capability': cluster,
                    'judge': {
                        'capability': cluster,
                        'question': question,
                        'question_id': question_id,
                    },
                })
        dataset = Dataset.from_list(raw_data)
        return dataset


def post_process_arenahard(completion):
    s = completion['prediction']
    if result := re.findall('\[\[([AB<>=]+)\]\]', s):
        return result[0]
    else:
        return None


def get_battles_from_judgment(judged_answers, references, WEIGHT=3):
    arena_hard_battles = pd.DataFrame()
    for judged_answer, reference in zip(judged_answers, references):
        output = {
            'model_a': reference['answer1'],
            'model_b': reference['answer2']
        }

        if judged_answer is not None:
            weight = 1
            if judged_answer == 'A=B':
                output['winner'] = 'tie'
            elif judged_answer == 'A>B':
                output['winner'] = 'model_a'
            elif judged_answer == 'A>>B':
                output['winner'] = 'model_a'
                weight = WEIGHT
            elif judged_answer == 'B>A':
                output['winner'] = 'model_b'
            elif judged_answer == 'B>>A':
                output['winner'] = 'model_b'
                weight = WEIGHT
            else:
                weight = 0
        else:
            weight = 0

        if weight:
            arena_hard_battles = pd.concat(
                [arena_hard_battles,
                 pd.DataFrame([output] * weight)])

    return arena_hard_battles


def compute_mle_elo(df, SCALE=400, BASE=10, INIT_RATING=1000):
    models = pd.concat([df['model_a'], df['model_b']]).unique()
    models = pd.Series(np.arange(len(models)), index=models)

    # duplicate battles
    df = pd.concat([df, df], ignore_index=True)
    p = len(models.index)
    n = df.shape[0]

    X = np.zeros([n, p])
    X[np.arange(n), models[df['model_a']]] = +math.log(BASE)
    X[np.arange(n), models[df['model_b']]] = -math.log(BASE)

    # one A win => two A win
    Y = np.zeros(n)
    Y[df['winner'] == 'model_a'] = 1.0

    # one tie => one A win + one B win
    # find tie + tie (both bad) index
    tie_idx = (df['winner'] == 'tie') | (df['winner'] == 'tie (bothbad)')
    tie_idx[len(tie_idx) // 2:] = False
    Y[tie_idx] = 1.0
    lr = LogisticRegression(
        fit_intercept=False, penalty=None, tol=1e-8
    )  # May need to set a small value when not use GPT4 as judge model
    lr.fit(X, Y)

    elo_scores = SCALE * lr.coef_[0] + INIT_RATING

    # set anchor as gpt4-0314 = 1000
    if 'gpt4-0314' in models.index:
        elo_scores += 1000 - elo_scores[models['gpt4-0314']]
    return pd.Series(elo_scores,
                     index=models.index).sort_values(ascending=False)


def get_bootstrap_result(battles, func_compute_elo, num_round):
    rows = []
    for i in range(num_round):
        rows.append(func_compute_elo(battles.sample(frac=1.0, replace=True)))
    df = pd.DataFrame(rows)
    return df[df.median().sort_values(ascending=False).index]


def preety_print_two_ratings(ratings_1, ratings_2, column_names):
    df = (pd.DataFrame(
        [[n, ratings_1[n], ratings_2[n]] for n in ratings_1.keys()],
        columns=['Model', column_names[0], column_names[1]],
    ).sort_values(column_names[0], ascending=False).reset_index(drop=True))
    df[column_names[0]] = (df[column_names[0]] + 0.5).astype(int)
    df[column_names[1]] = (df[column_names[1]] + 0.5).astype(int)
    df.index = df.index + 1
    return df


def predict_win_rate(elo_ratings, SCALE=400, BASE=10, INIT_RATING=1000):
    names = list(elo_ratings.keys())
    wins = defaultdict(lambda: defaultdict(lambda: 0))
    for a in names:
        for b in names:
            ea = 1 / (1 + BASE**((elo_ratings[b] - elo_ratings[a]) / SCALE))
            wins[a][b] = ea
            wins[b][a] = 1 - ea

    data = {
        a: [wins[a][b] if a != b else np.NAN for b in names]
        for a in names
    }

    df = pd.DataFrame(data, index=names)
    df.index.name = 'model_a'
    df.columns.name = 'model_b'
    return df.T


def get_win_rate_column(df, column, baseline='gpt4-0314'):
    to_dict = df[['model', column]].set_index('model').to_dict()[column]
    win_rate_table = predict_win_rate(to_dict)
    return win_rate_table[baseline].fillna(0.5).apply(
        lambda x: round(x * 100, 2))


@DICT_POSTPROCESSORS.register_module('arenahard')
def arenahard_postprocess(
    output: dict,
    output_path: str,
) -> dict:
    judged_answers, references = get_judgeanswer_and_reference(
        output, output_path, post_process_arenahard)

    if len(judged_answers) == 0:
        scores = None

    battles = get_battles_from_judgment(
        judged_answers,
        references,
    )

    bootstrap_online_elo = compute_mle_elo(battles)

    np.random.seed(42)
    bootstrap_elo_lu = get_bootstrap_result(battles, compute_mle_elo, 100)
    stats = pd.DataFrame()
    stats['results'] = None
    stats['results'] = stats['results'].astype('object')

    for i, model in enumerate(bootstrap_online_elo.index):
        assert model in bootstrap_elo_lu.columns
        stats.at[i, 'model'] = model
        stats.at[i, 'score'] = bootstrap_online_elo[model]
        # stats.at[i, 'lower'] = np.percentile(bootstrap_elo_lu[model], 2.5)
        # stats.at[i, 'upper'] = np.percentile(bootstrap_elo_lu[model], 97.5)
        # stats.at[i, 'results'] = bootstrap_elo_lu[model].tolist()

    stats['score'] = get_win_rate_column(stats, 'score', 'gpt4-0314').tolist()
    models = stats['model']
    scores = stats['score']
    if models[0] == 'gpt4-0314':
        score = scores[1]
    else:
        score = scores[0]

    results = {'score': score}
    results['details'] = output
    return results


@DICT_POSTPROCESSORS.register_module('arenahard_bradleyterry')
def arenahard_bradleyterry_postprocess(
    output: dict,
    output_path: str,
) -> dict:
    judged_answers, references = get_judgeanswer_and_reference(
        result=output,
        filename=output_path,
        post_process=post_process_arenahard,
    )

    if 'prediction1' not in references[0]:
        raise ValueError(
            'prediction1 not in references. Set `keep_predictions=True` for LMEvaluator in dataset config and retry.'
        )

    if 'prediction2' not in references[0]:
        raise ValueError(
            'prediction2 not in references. Set `keep_predictions=True` for LMEvaluator in dataset config and retry.'
        )

    results = {}
    matches = []
    for judged_answer, reference in zip(judged_answers, references):
        cur_dict = {}

        if judged_answer in ['A>>B', 'B<<A', 'A>B', 'B<A']:
            cur_dict['winner'] = 'model_a'
        elif judged_answer in ['A=B', 'B=A']:
            cur_dict['winner'] = 'tie'
        elif judged_answer in ['A<B', 'B>A', 'A<<B', 'B>>A']:
            cur_dict['winner'] = 'model_b'
        else:
            continue

        cur_dict['capability'] = reference['capability']
        cur_dict['model_a'] = reference['answer1']
        cur_dict['model_b'] = reference['answer2']
        cur_dict['prediction1'] = reference['prediction1']
        cur_dict['prediction2'] = reference['prediction2']

        matches.append(cur_dict)

    ### ---------- Add Style Metadata ---------- ###
    matches = get_element_counts(
        data=matches,
        column='prediction1',
        suffix='_a',
    )
    matches = get_element_counts(
        data=matches,
        column='prediction2',
        suffix='_b',
    )

    results['matches'] = matches
    # results["details"] = output

    return results
