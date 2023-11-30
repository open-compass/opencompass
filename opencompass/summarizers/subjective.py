import copy as cp
import io
import json
import math
import multiprocessing as mp
import os
import os.path as osp
import pickle
import random as rd
from collections import defaultdict
from datetime import datetime
from typing import List, Optional

try:
    import cv2
except ImportError:
    import traceback

    traceback.print_exc()
    raise ImportError(
        'Import cv2 failed. Please install it with '
        '"pip install opencv-python-headless" and try again.\n\n'
        'If the prompt `ImportError: libGL.so.1` appears,'
        ' you may consider one of the following two methods:\n'
        'Method 1 - Uninstall opencv and then install opencv-headless\n'
        'pip uninstall opencv-python; pip install opencv-python-headless\n\n'
        'Method 2: Install the missing dynamic link libraries\n'
        'sudo apt-get update; sudo apt-get install -y libgl1 libglib2.0-0')
import mmengine
import numpy as np
import pandas as pd
from mmengine import ConfigDict
from tabulate import tabulate
from tqdm import tqdm

from opencompass.utils import build_dataset_from_cfg, dataset_abbr_from_cfg


def dump(data, f):
    """Dump data to file."""

    def dump_pkl(data, pth):
        pickle.dump(data, open(pth, 'wb'))

    def dump_json(data, pth):
        json.dump(data, open(pth, 'w'), indent=4)

    def dump_jsonl(data, f):
        lines = [json.dumps(x, ensure_ascii=False) for x in data]
        with open(f, 'w', encoding='utf8') as fout:
            fout.write('\n'.join(lines))

    def dump_xlsx(data, f):
        data.to_excel(f, index=False)

    def dump_csv(data, f):
        data.to_csv(f, index=False)

    def dump_tsv(data, f):
        data.to_csv(f, sep='\t', index=False)

    handlers = dict(pkl=dump_pkl,
                    json=dump_json,
                    jsonl=dump_jsonl,
                    xlsx=dump_xlsx,
                    csv=dump_csv,
                    tsv=dump_tsv)
    suffix = f.split('.')[-1]
    return handlers[suffix](data, f)


def load(f):
    """Load data from file."""

    def load_pkl(pth):
        return pickle.load(open(pth, 'rb'))

    def load_json(pth):
        return json.load(open(pth, 'r', encoding='utf-8'))

    def load_jsonl(f):
        lines = open(f, encoding='utf-8').readlines()
        lines = [x.strip() for x in lines]
        if lines[-1] == '':
            lines = lines[:-1]
        data = [json.loads(x) for x in lines]
        return data

    def load_xlsx(f):
        return pd.read_excel(f)

    def load_csv(f):
        return pd.read_csv(f)

    def load_tsv(f):
        return pd.read_csv(f, sep='\t')

    handlers = dict(pkl=load_pkl,
                    json=load_json,
                    jsonl=load_jsonl,
                    xlsx=load_xlsx,
                    csv=load_csv,
                    tsv=load_tsv)
    suffix = f.split('.')[-1]
    return handlers[suffix](f)


def double_log(msg, fout=None):
    """Prints a message and optionally writes it to a file.

    Args:
        msg (str): The message to be printed and, if fout is provided,
            written to the file.
        fout (file, optional): A file object to write the message
            to (default is None).

    Returns:
        None
    """
    print(msg)
    if fout is not None:
        fout.write(str(msg) + '\n')
        fout.flush()


def stack_image(imgs, shape=(1, 3)):
    """Stacks a list of images into a grid.

    Args:
        imgs (list): A list of image arrays to be stacked.
        shape (tuple): A tuple specifying the grid shape
            (rows, columns) for the stacked images (default is (1, 3)).

    Returns:
        numpy.ndarray: The stacked image grid.
    """
    total_imgs = shape[0] * shape[1]
    assert len(imgs) <= total_imgs
    h, w, _ = imgs[0].shape
    imgs = [cv2.resize(im, dsize=(w, h)) for im in imgs]
    for i in range(total_imgs - len(imgs)):
        imgs.append(np.ones((h, w, 3)).astype(np.uint8) * 127)
    rows = []
    for i in range(shape[0]):
        if shape[1] == 1:
            rows.append(imgs[i])
        else:
            rows.append(np.hstack(imgs[i * shape[1]:(i + 1) * shape[1]]))
    if shape[0] == 1:
        return rows[0]
    else:
        return np.vstack(rows)


def simple_count(data_in, lang=None, capa=None):
    """Counts occurrences of outcomes (win, lose, both, neither) in a dataset.

    Args:
        data_in (dict): The input data containing 'A', 'B', 'extracted' fields.
        lang (str, optional): Filter by language (default is None).
        capa (str, optional): Filter by capability (default is None).

    Returns:
        dict: A dictionary containing outcome counts for each
            entry in 'A' and 'B'.
    """
    data = cp.deepcopy(data_in)
    if lang is not None and 'lang' in data:
        data = data[data['lang'] == lang]
    if capa is not None and 'capability' in data:
        flag = [(capa in x) for x in data['capability']]
        data = data[flag]

    A, B, ext = data['A'], data['B'], data['extracted']
    res = {}
    for a, b, choice in zip(A, B, ext):
        if a not in res:
            res[a] = defaultdict(lambda: 0)
        if b not in res:
            res[b] = defaultdict(lambda: 0)
        ans_map = dict(A=['win', 'lose'],
                       B=['lose', 'win'],
                       C=['both', 'both'],
                       D=['neither', 'neither'])
        ak, bk = ans_map[choice]
        res[a][ak] += 1
        res[b][bk] += 1
    return res


def calc_win_rate(data_copy, models, lang=None, capa=None):
    """Calculates win rates, tie rates, and loss rates between models based on
    given data.

    Args:
        data_copy (pd.DataFrame): The input data containing
            'A', 'B', 'extracted', 'lang', and 'capability' columns.
        models (list): List of model names to calculate rates for.
        lang (str, optional): Filter data by language (default is None).
        capa (str, optional): Filter data by capability (default is None).

    Returns:
        pd.DataFrame, pd.DataFrame: DataFrames containing win rates
            (cnt) and tie rates (ff) between models.
    """
    data = cp.deepcopy(data_copy)
    if lang is not None and 'lang' in data:
        data = data[data['lang'] == lang]
    if capa is not None and 'capability' in data:
        flag = [(capa in x) for x in data['capability']]
        data = data[flag]

    win = defaultdict(lambda: 0)
    tie = defaultdict(lambda: 0)
    lose = defaultdict(lambda: 0)

    for i in range(len(data)):
        v = data.iloc[i]
        o = v['extracted']
        key = v['A'] + ';' + v['B']

        if o == 'A':
            win[key] += 1
        if o == 'B':
            lose[key] += 1
        if o in ['C', 'D']:
            tie[key] += 1

    nmodel = len(models)
    cnt = pd.DataFrame({k: [0] * nmodel for k in models}, index=models)
    ff = pd.DataFrame({k: [0] * nmodel for k in models}, index=models)
    tot = pd.DataFrame({k: [0] * nmodel for k in models}, index=models)
    for i, k in enumerate(win):
        m1, m2 = k.split(';')
        cnt.at[m1, m2] += win[k]
        cnt.at[m2, m1] += lose[k]
        ff.at[m1, m2] += tie[k]
        ff.at[m2, m1] += tie[k]
        tot.at[m1, m2] += tie[k] + win[k] + lose[k]
        tot.at[m2, m1] += tie[k] + win[k] + lose[k]

    for m1 in models:
        for m2 in models:
            if tot.at[m1, m2]:
                cnt.at[m1, m2] /= tot.at[m1, m2]
                ff.at[m1, m2] /= tot.at[m1, m2]
    return cnt, ff


def find_inconsistent(data, vals=['A', 'B', 'C', 'D']):
    """Finds inconsistent data entries based on specified values.

    Args:
        data (pd.DataFrame): The input data containing
            'cmp_index' and 'extracted' columns.
        vals (list, optional): List of possible values
            (default is ['A', 'B', 'C', 'D']).

    Returns:
        pd.DataFrame, pd.DataFrame: DataFrames containing
            consistent (cons) and inconsistent (incons) data entries.
    """
    assert 'extracted' in data
    cons, incons = [], []
    pred_map = {x: y for x, y in zip(data['cmp_index'], data['extracted'])}
    for k in data['cmp_index']:
        parts = k.split(';')
        kct = ';'.join([parts[0], parts[2], parts[1]])
        if kct not in pred_map:
            cons.append(k)
            continue
        cons_tups = [(vals[0], vals[1]), (vals[1], vals[0]),
                     (vals[2], vals[2]), (vals[3], vals[3])]
        flag = True
        for tup in cons_tups:
            if pred_map[k] == tup[0] and pred_map[kct] == tup[1]:
                flag = False
                cons.append(k)
                break
        if flag:
            incons.append(k)
    cons, incons = data[data['cmp_index'].isin(cons)], data[
        data['cmp_index'].isin(incons)]
    return cons, incons


def extract_vispair(data, vals='ABCD', vispair=None):
    """Extracts specific data pairs and writes them to Excel files.

    Args:
        data (pd.DataFrame): The input data containing
            'A', 'B', and 'extracted' columns.
        vals (str, optional): A string of possible
            values (default is 'ABCD').
        vispair (tuple, optional): A tuple specifying the pair
            of values to extract (e.g., ('A', 'B')).

    Returns:
        None
    """
    assert vispair is not None
    ma, mb = vispair
    indices_map = defaultdict(list)
    lt = len(data)
    for i in range(lt):
        item = data.iloc[i]
        if (item['A'] == ma and item['B'] == mb
                and item['extracted'] == vals[0]):
            indices_map[f'{ma}_win_{mb}'].append(i)

        if (item['A'] == mb and item['B'] == ma
                and item['extracted'] == vals[1]):
            indices_map[f'{ma}_win_{mb}'].append(i)

        if (item['A'] == ma and item['B'] == mb
                and item['extracted'] == vals[1]):
            indices_map[f'{ma}_lose_{mb}'].append(i)

        if (item['A'] == mb and item['B'] == ma
                and item['extracted'] == vals[0]):
            indices_map[f'{ma}_lose_{mb}'].append(i)

        if (set([item['A'], item['B']]) == set([ma, mb])
                and item['extracted'] == vals[2]):
            indices_map[f'{ma}_both_{mb}'].append(i)

        if (set([item['A'], item['B']]) == set([ma, mb])
                and item['extracted'] == vals[3]):
            indices_map[f'{ma}_neither_{mb}'].append(i)

    for k in indices_map:
        data_sub = data.iloc[indices_map[k]]
        dump(data_sub, f'{k}.xlsx')


def get_shape(lt):
    """Calculates the shape (rows, columns) for a grid based on the number of
    elements.

    Args:
        lt (int): The total number of elements in the grid.

    Returns:
        tuple: A tuple containing the calculated number
            of rows and columns.
    """
    h = int(math.sqrt(lt))
    w = lt // h
    if h * w < lt:
        w += 1
    return h, w


def compute_elo_score(data,
                      K=32,
                      SCALE=400,
                      BASE=10,
                      INIT_RATING=1000,
                      seed=2680,
                      vals='ABCD'):
    """Computes Elo ratings for models based on provided data.

    Args:
        data (pd.DataFrame): The input data containing
            'A', 'B', and 'extracted' columns.
        K (float, optional): The K factor for Elo
            calculation (default is 32).
        SCALE (float, optional): The Elo scale factor (default is 400).
        BASE (float, optional): The Elo base factor (default is 10).
        INIT_RATING (float, optional): The initial rating
            for models (default is 1000).
        seed (int, optional): Random seed for shuffling
            battles (default is 2680).
        vals (str, optional): A string of possible values
            (default is 'ABCD').

    Returns:
        dict: A dictionary containing model ratings.
    """
    rating = defaultdict(lambda: INIT_RATING)
    battles = []
    lt = len(data)
    for i in range(lt):
        item = data.iloc[i]
        score_map = {vals[0]: 1, vals[1]: 0, vals[2]: 0.5, vals[3]: 0.5}
        score = score_map[
            item['extracted']] if item['extracted'] in score_map else 0.5
        battles.append((item['A'], item['B'], score))

    rd.seed(seed)
    rd.shuffle(battles)

    for m0, m1, v in battles:
        ra = rating[m0]
        rb = rating[m1]
        ea = 1 / (1 + BASE**((rb - ra) / SCALE))
        eb = 1 / (1 + BASE**((ra - rb) / SCALE))
        sa = v
        rating[m0] += K * (sa - ea)
        rating[m1] += K * (1 - sa - eb)
    return {k: v for k, v in rating.items()}


def compute_elo_score_pack(tup):
    return compute_elo_score(tup[0], seed=tup[1], vals=tup[2])


def mrlines(fname, sp='\n'):
    f = open(fname).read().split(sp)
    while f != [] and f[-1] == '':
        f = f[:-1]
    return f


def get_bootstrap_result(data,
                         num_round,
                         base_seed=1000,
                         num_thread=20,
                         vals='ABCD'):
    """Computes Elo scores with bootstrapping and returns the results as a
    DataFrame.

    Args:
        data (pd.DataFrame): The input data containing 'A', 'B',
            and 'extracted' columns.
        num_round (int): The number of bootstrap rounds to perform.
        base_seed (int, optional): The base seed for randomization
            (default is 1000).
        num_thread (int, optional): The number of threads to use
            for parallel processing (default is 20).
        vals (str, optional): A string of possible values
            (default is 'ABCD').

    Returns:
        pd.DataFrame: A DataFrame containing Elo scores for
            models based on bootstrapping.
    """
    rows = []
    tups = [(data, base_seed + i, vals) for i in range(num_round)]
    pool = mp.Pool(num_thread)
    rets = pool.map(compute_elo_score_pack, tups)
    for ret in rets:
        rows.append(ret)
    df = pd.DataFrame(rows)
    return df[df.median().sort_values(ascending=False).index]


def bootstrap_elo(data, num_round=1000, times=10, vals='ABCD'):
    """Computes Elo scores with bootstrapping over multiple runs and returns
    aggregated results.

    Args:
        data (pd.DataFrame): The input data containing 'A', 'B',
            and 'extracted' columns.
        num_round (int, optional): The number of bootstrap rounds
            to perform in each run (default is 1000).
        times (int, optional): The number of runs to perform
            (default is 10).
        vals (str, optional): A string of possible values
            (default is 'ABCD').

    Returns:
        pd.DataFrame: A DataFrame containing aggregated Elo
            scores with mean and standard deviation.
    """
    results = defaultdict(list)
    for i in tqdm(range(times)):
        bootstrap_elo_lu = get_bootstrap_result(data,
                                                num_round,
                                                base_seed=num_round * i,
                                                num_thread=20,
                                                vals=vals)
        bootstrap_lu_median = bootstrap_elo_lu.median().reset_index().set_axis(
            ['model', 'rating'], axis=1)
        for m, r in zip(bootstrap_lu_median['model'],
                        bootstrap_lu_median['rating']):
            results[m].append(r)
    res_dict = {}
    keys = list(results.keys())
    keys.sort()
    for k in keys:
        res_dict[k] = [np.mean(results[k]), np.std(results[k])]
    df = pd.DataFrame(res_dict, index=['elo_score [Mean]', 'elo_score [Std]'])
    return df


FONT_FILE = os.environ.get('FONT_FILE', None)


def match_answer(s):
    """Match the selected answer (A, B, C, or D) in a given string.

    Args:
        s (str): The input string to search for the selected answer.

    Returns:
        str or None: The matched answer ('A', 'B', 'C', or 'D')
            or None if not found.
    """

    def match_char(s, chars):
        cin = [c in s for c in chars]
        if sum(cin) == 1:
            return chars[cin.index(True)]
        else:
            return None

    lines = s.split('\n')
    for _, line in enumerate(lines):
        if line.startswith('选择：'):
            return match_char(line, 'ABCD')
    return None


def draw_heatmap(hmap, title):
    """Draw a heatmap using the given data.

    Args:
        hmap (pd.DataFrame): The data for the heatmap.
        title (str): The title for the heatmap.

    Returns:
        np.ndarray: An image of the heatmap.
    """
    from matplotlib import font_manager
    if FONT_FILE is None:
        fontP = font_manager.FontProperties()
    else:
        fontP = font_manager.FontProperties(fname=FONT_FILE)
    fontP.set_size(18)
    import matplotlib.pyplot as plt
    import seaborn as sns
    ax = sns.heatmap(hmap,
                     annot=True,
                     cmap='Blues',
                     annot_kws={'size': 35 / np.sqrt(len(hmap))})
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=12)
    plt.yticks(rotation=0)
    ax.xaxis.tick_top()  # x axis on top
    ax.xaxis.set_label_position('top')
    plt.title(title, color='Blue', fontproperties=fontP)
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    plt.close()
    buffer.seek(0)
    image_data = buffer.getvalue()
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    return image


def proc_capa(capas):
    capa_lists = [capa_str for capa_str in capas]
    capa_set = set(capa_lists)
    capa_set = list(capa_set)
    return capa_set


class SubjectiveSummarizer:
    """Do the subjectivity analyze based on evaluation results.

    Args:
        config (ConfigDict): The configuration object of the evaluation task.
            It's expected to be filled out at runtime.
        vispair (List[str], optional): List of
            two models to visualize.
        refm (str, optional): Reference model
            for win rate comparison.
        col_name (str): Name of the column
            containing evaluation results.
        fout (str): Output file name.
        ignore (str, optional): Ignore certain
            comparisons based on a file.
    """

    def __init__(
        self,
        config: ConfigDict,
        vispair: Optional[List[str]] = None,
        refm: Optional[str] = None,
        col_name: str = 'gpt4',
        fout: str = 'report.md',
        ignore: Optional[str] = None,
    ) -> None:
        self.tasks = []
        self.cfg = config
        self.vispair = vispair
        self.refm = refm
        self.col_name = col_name
        self.fout = fout
        self.ignore = ignore

    def summarize(self,
                  time_str: str = datetime.now().strftime('%Y%m%d_%H%M%S')):
        """Summarize the subjectivity analysis based on evaluation results.

        Args:
            time_str (str): Timestamp for file naming.

        Returns:
            pd.DataFrame: The summary results.
        """

        dataset_cfgs = self.cfg['datasets']
        eval_cfg = self.cfg['eval']
        work_dir = self.cfg['work_dir']
        self.work_dir = work_dir

        self.time_str = time_str
        output_path = osp.join(self.work_dir, 'summary',
                               f'summary_{self.time_str}.txt')
        output_dir = osp.join(osp.split(output_path)[0], f'{self.time_str}')
        mmengine.mkdir_or_exist(output_dir)
        fout = open(osp.join(output_dir, self.fout), 'w')
        results_folder = osp.join(work_dir, 'results')
        data_list = []
        for subdir in os.listdir(results_folder):
            subdir_path = os.path.join(results_folder, subdir)
            if os.path.isdir(subdir_path):
                model1, model2 = subdir.split('_')
                for dataset in dataset_cfgs:
                    origin_dataset = build_dataset_from_cfg(dataset)
                    dataset_abbr = dataset_abbr_from_cfg(dataset)
                    filepath = os.path.join(subdir_path,
                                            dataset_abbr + '.json')
                    result = mmengine.load(filepath)
                    if eval_cfg['partitioner']['mode'] == 'all':
                        for key, value in result.items():
                            prediction = value.get('prediction', None)
                            q_index = origin_dataset.test[int(key) % len(
                                origin_dataset.test)]['index']
                            cmp_index = f'{q_index};{model1};{model2}'
                            data_list.append(
                                [cmp_index, model1, model2, prediction])

        data = pd.DataFrame(data_list, columns=['cmp_index', 'A', 'B', 'gpt4'])
        meta = pd.read_excel(
            osp.join(dataset_cfgs[0]['path'],
                     dataset_cfgs[0]['name'] + '.xlsx'))

        if self.ignore is not None:
            q_index = [x.split(';')[0] for x in data['cmp_index']]
            to_ignore = set(mrlines(self.ignore))
            flag = [x not in to_ignore for x in q_index]
            data = data[flag]

        double_log('# Subjective Analysis', fout)
        capas = proc_capa(meta['capability'])
        capa_map = {i: c for i, c in zip(meta['index'], meta['capability'])}

        nonem = [x != 'EM' for x in data[self.col_name]]
        double_log(
            f'A total of {len(data)} comparisons, of which {sum(nonem)} '
            f'comparisons are meaningful (A / B answers inconsistent)', fout)
        data = data[nonem]

        data['capability'] = [
            capa_map[str(i).split(';')[0]] for i in data['cmp_index']
        ]
        data['extracted'] = [match_answer(ans) for ans in data[self.col_name]]

        succeed = [not pd.isna(x) for x in data['extracted']]
        succeed_rate = np.mean(succeed)
        double_log(
            f'A total of {len(succeed)} answer comparisons, successfully '
            f'extracted {sum(succeed)} answers from GPT-4 replies, with '
            f'an extraction success rate of {succeed_rate * 100:.2f}%', fout)
        data = data[succeed]

        cons, incons = find_inconsistent(data, 'ABCD')
        if len(cons) != len(data):
            double_log(
                f'A total of {len(data)} answer comparisons, {len(cons)} '
                f'pairs (A vs. B <-> B vs. A) are consistent，consistent '
                f'rate is {len(cons) / len(data) * 100:.2f}%', fout)

        dump(cons, osp.join(output_dir, 'consistent_cmp.xlsx'))
        dump(incons, osp.join(output_dir, 'inconsistent_cmp.xlsx'))

        data = cons
        if self.vispair is not None and len(self.vispair) == 2:
            extract_vispair(data, vispair=self.vispair)

        data['lang'] = [x.split('-')[0] for x in data['cmp_index']]
        langs = [None, 'cn', 'en']
        return self.analyze(data, self.refm, langs, capas, fout)

    def analyze(self, data, refm, langs, capas, fout):
        """Do the subjectivity analysis based on evaluation results.

        Args:
            data (pd.DataFrame): The evaluation data.
            refm (str): Reference model for win rate comparison.
            langs (List[str]): List of languages to analyze.
            capas (List[str]): List of capabilities to analyze.
            fout (str): Output file name.

        Returns:
            None
        """
        output_path = osp.join(self.work_dir, 'summary',
                               f'summary_{self.time_str}.txt')
        output_dir = osp.join(osp.split(output_path)[0], f'{self.time_str}')
        mmengine.mkdir_or_exist(output_dir)

        stats = defaultdict(list)
        scores = defaultdict(list)

        dim_key = 'Dimension \\ Stat [W / T / L / NB]'
        scores_dim_key = 'Dimension \\ Score'

        for lang in langs:
            name = (lang.upper() if lang is not None else 'Overall')
            stats[dim_key].append(f'LANG: {name}')
            scores[scores_dim_key].append(f'LANG: {name}')

            count_stat = simple_count(data, lang=lang)
            if count_stat == {}:
                for k, v in stats.items():
                    if k != dim_key:
                        v.append('N/A')
                for k, v in scores.items():
                    if k != scores_dim_key:
                        v.append('N/A')

            for k in count_stat:
                stat = count_stat[k]
                winr = stat['win'] / sum(stat.values())
                tier = (stat['both'] + stat['neither']) / sum(stat.values())
                loser = stat['lose'] / sum(stat.values())
                not_bad = (stat['win'] + stat['both']) / sum(stat.values())
                msg = f'{winr * 100:.1f}% / {tier * 100:.1f}% / {loser * 100:.1f}% / {not_bad * 100:.1f}%'  # noqa
                stats[k].append(msg)
                score = 3 * stat['win'] + stat['both'] - stat[
                    'neither'] - 3 * stat['lose']
                scores[k].append(score)
        for capa in capas:
            stats[dim_key].append(f'CAPA: {capa}')
            scores[scores_dim_key].append(f'CAPA: {capa}')
            count_stat = simple_count(data, capa=capa)
            if count_stat == {}:
                for k, v in stats.items():
                    if k != dim_key:
                        v.append('N/A')
                for k, v in scores.items():
                    if k != scores_dim_key:
                        v.append('N/A')

            for k in count_stat:
                stat = count_stat[k]
                winr = stat['win'] / sum(stat.values())
                tier = (stat['both'] + stat['neither']) / sum(stat.values())
                loser = stat['lose'] / sum(stat.values())
                not_bad = (stat['win'] + stat['both']) / sum(stat.values())
                msg = f'{winr * 100:.1f}% / {tier * 100:.1f}% / {loser * 100:.1f}% / {not_bad * 100:.1f}%'  # noqa
                stats[k].append(msg)
                score = 3 * stat['win'] + stat['both'] - stat[
                    'neither'] - 3 * stat['lose']
                scores[k].append(score)
        double_log(
            '### Basic statistics (4 stats: win / tie / lose / not bad)', fout)
        all_models = list(stats.keys())
        all_models.remove(dim_key)

        table_width = 3
        num_tables = len(all_models) // table_width + (
            len(all_models) % table_width != 0)
        for i in range(num_tables):
            cur_keys = [dim_key
                        ] + all_models[i * table_width:(i + 1) * table_width]
            sub_stats = {k: stats[k] for k in cur_keys}
            double_log(tabulate(sub_stats, headers='keys', tablefmt='github'),
                       fout)

        image_url1 = 'by_capa.png'
        image_url2 = 'by_lang.png'
        double_log(
            f'\n\n![Capabilities Dimension '
            f'Classification Result]({image_url1})'
            f'\n\n![Language Classification  Result]({image_url2})', fout)

        double_log(
            '\n\n### Model scores (base score is 0, win +3,'
            ' both +1, neither -1, lose -3)', fout)
        double_log(tabulate(scores, headers='keys', tablefmt='github'), fout)

        double_log('### Bootstrap ELO, Median of n=1000 times ', fout)
        elo_table = bootstrap_elo(data)
        double_log(tabulate(elo_table, headers='keys', tablefmt='github'),
                   fout)

        models = list(count_stat.keys())
        models.sort()

        images = []
        for lang in langs:
            wr, dr = calc_win_rate(data, models, lang=lang)
            lang_name = lang.upper() if lang is not None else 'Overall'

            wr_table = defaultdict(list)
            if refm is not None:
                for m in models:
                    if m == refm:
                        continue
                    wr_table['model'].append(m)
                    wr_table['win_rate'].append(wr.at[m, refm])
                    wr_table['draw_rate'].append(dr.at[m, refm])
                    wr_table['win + draw'].append(dr.at[m, refm] +
                                                  wr.at[m, refm])
                double_log(
                    f'By language {lang_name}, calculate '
                    f'the win rate against {refm}:', fout)
                double_log(
                    tabulate(wr_table, headers='keys', tablefmt='github'),
                    fout)

            im = draw_heatmap(
                wr, f'Language: {lang if lang is not None else "All"}')
            images.append(im)
        image = stack_image(images, shape=(1, 3))
        cv2.imwrite(osp.join(output_dir, 'by_lang.png'), image)

        images = []
        for capa in capas:
            wr, dr = calc_win_rate(data, models, capa=capa)

            wr_table = defaultdict(list)
            if refm is not None:
                for m in models:
                    if m == refm:
                        continue
                    wr_table['model'].append(m)
                    wr_table['win_rate'].append(wr.at[m, refm])
                    wr_table['draw_rate'].append(dr.at[m, refm])
                    wr_table['win + draw'].append(dr.at[m, refm] +
                                                  wr.at[m, refm])
                double_log(
                    f'By capability {capa}, calculate the '
                    f'win rate against {refm}:', fout)
                double_log(
                    tabulate(wr_table, headers='keys', tablefmt='github'),
                    fout)

            im = draw_heatmap(wr, f'Capability: {capa}')
            images.append(im)

        lt = len(capas)
        h, w = get_shape(lt)
        image = stack_image(images, shape=(h, w))
        cv2.imwrite(osp.join(output_dir, 'by_capa.png'), image)
        dump(data, osp.join(output_dir, 'tmp.xlsx'))
        fout.close()
