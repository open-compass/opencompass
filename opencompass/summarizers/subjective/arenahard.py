# flake8: noqa
# yapf: disable
import argparse
import datetime
import json
import math
import os
import os.path as osp
import re
from collections import defaultdict
from datetime import datetime
from glob import glob
from itertools import product

import mmengine
import numpy as np
#import plotly.express as px
import pandas as pd
import tiktoken
from mmengine import ConfigDict
from sklearn.linear_model import LogisticRegression
from tabulate import tabulate
from tqdm import tqdm

from opencompass.partitioners.sub_naive import remove_duplicate_pairs
from opencompass.utils import dataset_abbr_from_cfg, model_abbr_from_cfg

from .utils import get_outdir


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
    tie_idx[len(tie_idx)//2:] = False
    Y[tie_idx] = 1.0
    lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-8) # May need to set a small value when not use GPT4 as judge model
    lr.fit(X,Y)

    elo_scores = SCALE * lr.coef_[0] + INIT_RATING

    # set anchor as gpt4-0314 = 1000
    if 'gpt4-0314' in models.index:
        elo_scores += 1000 - elo_scores[models['gpt4-0314']]
    return pd.Series(elo_scores, index = models.index).sort_values(ascending=False)


def get_bootstrap_result(battles, func_compute_elo, num_round):
    rows = []
    for i in tqdm(range(num_round), desc='bootstrap'):
        rows.append(func_compute_elo(battles.sample(frac=1.0, replace=True)))
    df = pd.DataFrame(rows)
    return df[df.median().sort_values(ascending=False).index]


def preety_print_two_ratings(ratings_1, ratings_2, column_names):
    df = pd.DataFrame([
        [n, ratings_1[n], ratings_2[n]] for n in ratings_1.keys()
    ], columns=['Model', column_names[0], column_names[1]]).sort_values(column_names[0], ascending=False).reset_index(drop=True)
    df[column_names[0]] = (df[column_names[0]] + 0.5).astype(int)
    df[column_names[1]] = (df[column_names[1]] + 0.5).astype(int)
    df.index = df.index + 1
    return df


def visualize_bootstrap_scores(df, title):
    bars = pd.DataFrame(dict(
        lower = df.quantile(.025),
        rating = df.quantile(.5),
        upper = df.quantile(.975))).reset_index(names='model').sort_values('rating', ascending=False)
    bars['error_y'] = bars['upper'] - bars['rating']
    bars['error_y_minus'] = bars['rating'] - bars['lower']
    bars['rating_rounded'] = np.round(bars['rating'], 2)
    fig = px.scatter(bars, x='model', y='rating', error_y='error_y',
                     error_y_minus='error_y_minus', text='rating_rounded',
                     title=title)
    fig.update_layout(xaxis_title='Model', yaxis_title='Rating',
                      height=600)
    return fig


def predict_win_rate(elo_ratings, SCALE=400, BASE=10, INIT_RATING=1000):
    names = sorted(list(elo_ratings.keys()))
    wins = defaultdict(lambda: defaultdict(lambda: 0))
    for a in names:
        for b in names:
            ea = 1 / (1 + BASE ** ((elo_ratings[b] - elo_ratings[a]) / SCALE))
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


def model_abbr_from_cfg_used_in_summarizer(model):
    if model.get('summarizer_abbr', None):
        return model['summarizer_abbr']
    else:
        return model_abbr_from_cfg(model)

def post_process_compass_arena(s):
    if result := re.findall('\[\[([AB<>=]+)\]\]', s):
        return result[0]
    else:
        return None

def get_win_rate_column(df, column, baseline='gpt4-0314'):
    to_dict = df[['model', column]].set_index('model').to_dict()[column]
    win_rate_table = predict_win_rate(to_dict)
    return win_rate_table[baseline].fillna(0.5).apply(lambda x: round(x * 100, 2))


def load_model_preds(filename):
    root, ext = osp.splitext(filename)
    partial_filename = root + '_0' + ext
    if osp.exists(osp.realpath(filename)):
        preds = mmengine.load(filename)
        pred_strs = [
            preds[str(i)]['prediction'] for i in range(len(preds))
        ]
    else:
        filename = partial_filename
        pred_strs = []
        i = 1
        while osp.exists(osp.realpath(filename)):
            preds = mmengine.load(filename)
            filename = root + f'_{i}' + ext
            i += 1
            pred_strs += [
                preds[str(i)]['prediction'] for i in range(len(preds))
            ]
    return pred_strs

def get_battles_from_judgment(dataset, subdir_path, post_process, WEIGHT=3):
    arena_hard_battles = pd.DataFrame()
    dataset_abbr = dataset_abbr_from_cfg(dataset)
    filename = osp.join(subdir_path, dataset_abbr + '.json')
    partial_filename = osp.join(subdir_path, dataset_abbr + '_0.json')
    if osp.exists(osp.realpath(filename)):
        result = mmengine.load(filename)
    elif osp.exists(osp.realpath(partial_filename)):
        filename = partial_filename
        result = {}
        i = 1
        partial_dict_flag = 0
        while osp.exists(osp.realpath(filename)):
            res = mmengine.load(filename)
            for k, v in res.items():
                result[partial_dict_flag] = v
                partial_dict_flag += 1
            filename = osp.join(subdir_path,
                                dataset_abbr + '_' + str(i) + '.json')
            i += 1
    else:
        result = {}

    if len(result) == 0:
        print('*' * 100)
        print('There are no results for ' + filename + ' or ' +
              partial_filename)
        print('*' * 100)
        assert len(result) > 0

    judged_answers = []
    references = []
    for k, v in result.items():

        output = {
                'model_a': v['gold']['answer1'],
                'model_b': v['gold']['answer2']}

        processed_judge = post_process(v['prediction'])
        if processed_judge is not None:
            weight = 1
            if processed_judge == 'A=B':
                output['winner'] = 'tie'
            elif processed_judge == 'A>B':
                output['winner'] = 'model_a'
            elif processed_judge == 'A>>B':
                output['winner'] = 'model_a'
                weight = WEIGHT
            elif processed_judge == 'B>A':
                output['winner'] = 'model_b'
            elif processed_judge == 'B>>A':
                output['winner'] = 'model_b'
                weight = WEIGHT
            else:
                weight = 0
        else:
            weight = 0

        if weight:
            arena_hard_battles = pd.concat([arena_hard_battles, pd.DataFrame([output] * weight)])

    return arena_hard_battles

class ArenaHardSummarizer:
    """Do the subjectivity analyze based on evaluation results.

    Args:
        config (ConfigDict): The configuration object of the evaluation task.
            It's expected to be filled out at runtime.
    """

    def __init__(self,
                 config: ConfigDict,
                 judge_type='general',
                 check_pos_bias=True,
                 summary_type='single') -> None:
        self.tasks = []
        self.cfg = config
        self.base_models = self.cfg['datasets'][0]['base_models']
        self.compare_models = self.cfg['eval']['partitioner']['models']
        self.judge_models = self.cfg.get('judge_models', None)
        self.meta_judge_model = self.cfg.eval.partitioner.get('meta_judge_model', None)
        self.judge_type = judge_type
        assert self.judge_type in ['general']
        self.judge_map = {'general': post_process_compass_arena}
        self.judge_function = self.judge_map[self.judge_type]
        self.check_pos_bias = check_pos_bias
        self.summary_type = summary_type

    def get_score(self, time_str):
        output_dir, results_folder = get_outdir(self.cfg, time_str)
        model_combinations = list(product(self.base_models, self.compare_models))
        unique_combinations = remove_duplicate_pairs([combo for combo in model_combinations if combo[0] != combo[1]])

        if self.meta_judge_model is not None:
            self.judge_models.append(self.meta_judge_model)

        all_scores = {}

        for idx, judge_model_cfg in enumerate(self.judge_models):
            score_by_judgemodel = {}
            judge_model = model_abbr_from_cfg(judge_model_cfg)
            for dataset in self.cfg['datasets']:
                dataset_abbr = dataset_abbr_from_cfg(dataset)
                battles = pd.DataFrame()
                print('Turning judgment results into battles...')
                for model_pair in unique_combinations:
                    model1 = model_pair[0]['abbr'] # base model, in ArenaHard it is gpt4-0314
                    model2 = model_pair[1]['abbr'] # compare model, your models
                    if idx == len(self.judge_models):
                        subdir = model1 + '_' + model2 + '_summarized-by--' + judge_model
                    else:
                        subdir = model1 + '_' + model2 + '_judged-by--' + judge_model
                    subdir_path = os.path.join(results_folder, subdir)
                    dataset_abbr = dataset_abbr_from_cfg(dataset)
                    filename = osp.realpath(osp.join(subdir_path, dataset_abbr + '.json'))
                    partial_filename = osp.realpath(osp.join(subdir_path, dataset_abbr + '_0.json'))
                    if not osp.exists(osp.realpath(filename)) and not osp.exists(osp.realpath(partial_filename)):
                        score_by_judgemodel[model2] = None
                        print(subdir_path + ' is not exist! please check!')
                        continue

                    new_battle = get_battles_from_judgment(dataset, subdir_path, self.judge_function)
                    battles = pd.concat([battles, new_battle], ignore_index=True)
                battles.to_json(os.path.join(output_dir,'arena_hard_battles_judged-by--'+ judge_model+'.jsonl'), lines=True, orient='records')

                bootstrap_online_elo = compute_mle_elo(battles)

                np.random.seed(42)
                bootstrap_elo_lu = get_bootstrap_result(battles, compute_mle_elo, 100)
                bootstrap_elo_lu.to_json(os.path.join(output_dir,'arena_hard_bootstrapping_results_judged-by--'+ judge_model+'.jsonl'), lines=True, orient='records')

                stats = pd.DataFrame()
                stats['results'] = None
                stats['results'] = stats['results'].astype('object')

                for i, model in enumerate(bootstrap_online_elo.index):
                    assert model in bootstrap_elo_lu.columns

                    stats.at[i, 'model'] = model
                    stats.at[i, 'score'] = bootstrap_online_elo[model]
                    stats.at[i, 'lower'] = np.percentile(bootstrap_elo_lu[model], 2.5)
                    stats.at[i, 'upper'] = np.percentile(bootstrap_elo_lu[model], 97.5)
                    if model == model1:
                        if model1 == 'gpt4-0314':
                            stats.at[i, 'avg_tokens'] = 423
                        else:
                            stats.at[i, 'avg_tokens'] = 0 # Not expected model
                    else:
                        file_name = os.path.join(output_dir.split('summary')[0], 'predictions', model, dataset_abbr+'.json')
                        model_preds = load_model_preds(file_name)
                        pred_length = 0
                        for model_pred in model_preds:
                            pred_length += len(tiktoken.encoding_for_model('gpt-3.5-turbo').encode(model_pred, disallowed_special=()))
                        pred_length /= len(model_preds)
                        stats.at[i, 'avg_tokens'] = pred_length
                    stats.at[i, 'results'] = bootstrap_elo_lu[model].tolist()
                stats.sort_values(by='model', inplace=True)
                stats['score'] = get_win_rate_column(stats, 'score', model1).tolist()
                stats['lower'] = get_win_rate_column(stats, 'lower', model1).tolist()
                stats['upper'] = get_win_rate_column(stats, 'upper', model1).tolist()
                decimal = 1
                stats.sort_values(by='score', ascending=False, inplace=True)
                for _, row in stats.iterrows():
                    interval = str((round(row['lower'] - row['score'], decimal), round(row['upper'] - row['score'], decimal)))
                    print(f"{row['model'] : <30} | score: {round(row['score'], decimal) : ^5} | 95% CI: {interval : ^12} | average #tokens: {int(row['avg_tokens'])}")
                    if row['model'] != model1:
                        score_by_judgemodel[row['model']] = {'score': row['score']}
                stats.to_json(os.path.join(output_dir,'arena_hard_leaderboard_judged-by--'+judge_model+'.json'), orient='records', indent=4)
                stats.to_csv(os.path.join(output_dir,'arena_hard_leaderboard_judged-by--'+judge_model+'.csv'))
            all_scores[judge_model] = score_by_judgemodel
        return {'ArenaHard': all_scores}

    def summarize(
            self,
            time_str: str = datetime.now().strftime('%Y%m%d_%H%M%S'),
    ):
        """Summarize the subjectivity analysis based on evaluation results.

        Args:
            time_str (str): Timestamp for file naming.

        Returns:
            pd.DataFrame: The summary results.
        """
        return self.get_score(time_str)
