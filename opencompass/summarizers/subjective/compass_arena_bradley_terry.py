# flake8: noqa
import functools
import getpass
import json
import math
import multiprocessing as mp
import os
import os.path as osp
from collections import defaultdict
from datetime import datetime
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import mmengine
import numpy as np
import pandas as pd
import tabulate
from mmengine import ConfigDict
from scipy.optimize import minimize
from scipy.special import expit
from tqdm import tqdm

from opencompass.summarizers import DefaultSubjectiveSummarizer
from opencompass.summarizers.default_subjective import \
    model_abbr_from_cfg_used_in_summarizer
from opencompass.utils import (LarkReporter, dataset_abbr_from_cfg,
                               get_infer_output_path, get_logger,
                               model_abbr_from_cfg)
from opencompass.utils.prompt import get_prompt_hash

STYLE_CONTROL_VARIABLES_V1 = [
    'sum_assistant_tokens',
    'header_count',
    'list_count',
    'bold_count',
]

EXTRA_CONTROL_VARIABLES = []


def get_matchups_models(df):
    n_rows = len(df)
    model_indices, models = pd.factorize(
        pd.concat([df['model_a'], df['model_b']]))
    matchups = np.column_stack(
        [model_indices[:n_rows], model_indices[n_rows:]])
    return matchups, models.to_list()


def preprocess_for_elo(df):
    """
    in Elo we want numpy arrays for matchups and outcomes
      matchups: int32 (N,2)  contains model ids for the competitors in a match
      outcomes: float64 (N,) contains 1.0, 0.5, or 0.0 representing win, tie, or loss for model_a
    """
    matchups, models = get_matchups_models(df)
    outcomes = np.full(len(df), 0.5)
    outcomes[df['winner'] == 'model_a'] = 1.0
    outcomes[df['winner'] == 'model_b'] = 0.0
    return matchups, outcomes, models


def preprocess_for_bt(df):
    """In BT we only need the unique (matchup,outcome) sets along with the
    weights of how often they occur."""
    n_rows = len(df)
    # the 3 columns of schedule represent: model_a id, model_b id, outcome_id
    schedule = np.full((n_rows, 3), fill_value=1, dtype=np.int32)
    # set the two model cols by mapping the model names to their int ids
    schedule[:, [0, 1]], models = get_matchups_models(df)
    # map outcomes to integers (must be same dtype as model ids so it can be in the same array)
    # model_a win -> 2, tie -> 1 (prefilled by default), model_b win -> 0
    schedule[df['winner'] == 'model_a', 2] = 2
    schedule[df['winner'] == 'model_b', 2] = 0
    # count the number of occurrences of each observed result
    matchups_outcomes, weights = np.unique(schedule,
                                           return_counts=True,
                                           axis=0)
    matchups = matchups_outcomes[:, [0, 1]]
    # map 2 -> 1.0, 1 -> 0.5, 0 -> 0.0 which will be used as labels during optimization
    outcomes = matchups_outcomes[:, 2].astype(np.float64) / 2.0
    weights = weights.astype(np.float64)
    # each possible result is weighted according to number of times it occurred in the dataset
    return matchups, outcomes, models, weights


def preprocess_for_style(
    df,
    apply_ratio: List[int] = None,
    style_variables: List[str] = STYLE_CONTROL_VARIABLES_V1,
    control_variables: List[str] = EXTRA_CONTROL_VARIABLES,
    style_var_suffixes: List[str] = None,
    add_one: bool = True,
    normalize_style_features: bool = True,
):
    matchups, outcomes, models = preprocess_for_elo(
        df)  # this can use the same preprocessing as Elo

    n = matchups.shape[0]
    style_k = int(len(style_variables))

    if control_variables is not None:
        control_k = int(len(control_variables))
    else:
        control_k = 0

    if apply_ratio == None:
        apply_ratio = np.repeat(1, style_k)

    def extract_feature(x, feature):
        val = x[feature]
        if isinstance(val, int):
            return val
        else:
            return sum(val.values())

    ## Style variables
    if style_var_suffixes is None:
        style_var_suffixes = ['_a', '_b']

    style_vector = np.zeros(shape=(2 * style_k, n), dtype=np.int32)
    for idx1, model_suffix in enumerate(style_var_suffixes):
        for idx, element in enumerate(style_variables):
            style_vector[idx + (idx1 * style_k), :] = df.conv_metadata.map(
                partial(extract_feature,
                        feature=f'{element}{model_suffix}')).values

    style_vector = np.ascontiguousarray(style_vector)

    style_diff = (style_vector[:style_k] -
                  style_vector[style_k:]).astype(float)
    style_sum = (style_vector[:style_k] + style_vector[style_k:]).astype(float)

    # Add one to prevent division by zero
    if add_one:
        style_sum = style_sum + np.ones(style_diff.shape)

    apply_ratio = np.flatnonzero(apply_ratio)

    # Apply ratio where necessary (length, etc)
    style_diff[apply_ratio] /= style_sum[apply_ratio]

    style_mean = np.mean(style_diff, axis=1)

    if normalize_style_features:
        style_std = np.std(style_diff, axis=1)

        # # features = normalize(style_diff)
        style_features = ((style_diff - style_mean[:, np.newaxis]) /
                          style_std[:, np.newaxis]).T
    else:
        style_features = style_diff.T

    ## Other control variables
    if control_k > 0:
        control_vector = np.zeros(shape=(control_k, n), dtype=np.int32)
        for idx, element in enumerate(control_variables):
            control_vector[idx, :] = df[element]

        control_vector = np.ascontiguousarray(control_vector).astype(float)

        control_features = control_vector.T

        # combine style and other control features
        features = np.hstack([style_features, control_features])
    else:
        features = style_features

    return matchups, features, outcomes, models


def fit_vectorized_elo(
    matchups,
    outcomes,
    sample_indices,
    num_models: int,
    k: float = 4.0,
    base: float = 10.0,
    init_rating: float = 1000.0,
    scale: float = 400.0,
):
    """Fit multiple sets of Elo ratings on different samples of the data at the
    same time."""
    alpha = math.log(base) / scale
    num_samples = sample_indices.shape[1]
    ratings = np.zeros(shape=(num_samples, num_models), dtype=np.float64)
    # iterate over the rows of sample_indices, each column is an index into a match in the input arrays
    sample_range = np.arange(num_samples)
    for matchup_indices in sample_indices:
        model_a_indices = matchups[matchup_indices, 0]
        model_b_indices = matchups[matchup_indices, 1]
        model_a_ratings = ratings[sample_range, model_a_indices]
        model_b_ratings = ratings[sample_range, model_b_indices]
        sample_outcomes = outcomes[matchup_indices]
        probs = expit(alpha * (model_a_ratings - model_b_ratings))
        updates = k * (sample_outcomes - probs)
        ratings[sample_range, model_a_indices] += updates
        ratings[sample_range, model_b_indices] -= updates
    return ratings + init_rating


def compute_elo(
    df,
    k: float = 4.0,
    base: float = 10.0,
    init_rating: float = 1000.0,
    scale: float = 400.0,
):
    matchups, outcomes, models = preprocess_for_elo(df)
    alpha = math.log(base) / scale
    ratings = np.full(shape=(len(models), ), fill_value=init_rating)

    for (model_a_idx, model_b_idx), outcome in zip(matchups, outcomes):
        prob = 1.0 / (1.0 +
                      math.exp(alpha *
                               (ratings[model_b_idx] - ratings[model_a_idx])))
        update = k * (outcome - prob)
        ratings[model_a_idx] += update
        ratings[model_b_idx] -= update

    return {model: ratings[idx] for idx, model in enumerate(models)}


def compute_bootstrap_elo(
    df,
    num_round: int = 100,
    k: float = 4.0,
    base: float = 10.0,
    init_rating: float = 1000.0,
    scale: float = 400.0,
):
    matchups, outcomes, models = preprocess_for_elo(df)
    sample_indices = np.random.randint(low=0,
                                       high=len(df),
                                       size=(len(df), num_round))
    ratings = fit_vectorized_elo(matchups, outcomes, sample_indices,
                                 len(models), k, base, init_rating, scale)
    df = pd.DataFrame(data=ratings, columns=models)
    return df[df.median().sort_values(ascending=False).index]


def bt_loss_and_grad(ratings, matchups, outcomes, weights, alpha=1.0):
    matchup_ratings = ratings[matchups]
    logits = alpha * (matchup_ratings[:, 0] - matchup_ratings[:, 1])
    probs = expit(logits)
    # this form naturally counts a draw as half a win and half a loss
    loss = -((np.log(probs) * outcomes + np.log(1.0 - probs) *
              (1.0 - outcomes)) * weights).sum()
    matchups_grads = -alpha * (outcomes - probs) * weights
    model_grad = np.zeros_like(ratings)
    # aggregate gradients at the model level using the indices in matchups
    np.add.at(
        model_grad,
        matchups[:, [0, 1]],
        matchups_grads[:, None] * np.array([1.0, -1.0], dtype=np.float64),
    )
    return loss, model_grad


def fit_bt(matchups, outcomes, weights, n_models, alpha, tol=1e-6):
    initial_ratings = np.zeros(n_models, dtype=np.float64)
    result = minimize(
        fun=bt_loss_and_grad,
        x0=initial_ratings,
        args=(matchups, outcomes, weights, alpha),
        jac=True,
        method='L-BFGS-B',
        options={
            'disp': False,
            'maxiter': 100,
            'gtol': tol
        },
    )
    return result['x']


def scale_and_offset(
    ratings,
    models,
    scale: float = 400.0,
    init_rating: float = 1000.0,
    baseline_model: str = None,
    baseline_rating: float = 1000.0,
):
    """Convert ratings from the natural scale to the Elo rating scale with an
    anchored baseline."""
    scaled_ratings = (ratings * scale) + init_rating

    if baseline_model is not None:
        if baseline_model in models:
            baseline_idx = models.index(baseline_model)
            scaled_ratings += baseline_rating - scaled_ratings[...,
                                                               [baseline_idx]]

    return scaled_ratings


def compute_bt(
    df,
    base: float = 10.0,
    scale: float = 400.0,
    init_rating: float = 1000.0,
    baseline_model: str = None,
    baseline_rating: float = 1000.0,
    tol: float = 1e-6,
):
    matchups, outcomes, models, weights = preprocess_for_bt(df)
    ratings = fit_bt(matchups, outcomes, weights, len(models), math.log(base),
                     tol)

    scaled_ratings = scale_and_offset(
        ratings=ratings,
        models=models,
        scale=scale,
        init_rating=init_rating,
        baseline_model=baseline_model,
        baseline_rating=baseline_rating,
    )

    return pd.Series(scaled_ratings, index=models).sort_values(ascending=False)


def compute_bootstrap_bt(
    battles,
    num_round: int,
    base: float = 10.0,
    scale: float = 400.0,
    init_rating: float = 1000.0,
    baseline_model: str = None,
    baseline_rating: float = 1000.0,
    tol: float = 1e-6,
    num_cpu: int = None,
):
    matchups, outcomes, models, weights = preprocess_for_bt(battles)
    # bootstrap sample the unique outcomes and their counts directly using the multinomial distribution
    rng = np.random.default_rng(seed=0)
    idxs = rng.multinomial(n=len(battles),
                           pvals=weights / weights.sum(),
                           size=(num_round))
    # only the distribution over their occurrence counts changes between samples (and it can be 0)
    boot_weights = idxs.astype(np.float64) / len(battles)

    # the only thing different across samples is the distribution of weights
    bt_fn = partial(fit_bt,
                    matchups,
                    outcomes,
                    n_models=len(models),
                    alpha=np.log(base),
                    tol=tol)
    with mp.Pool(num_cpu if num_cpu else os.cpu_count() - 1) as pool:
        results = list(
            tqdm(pool.imap_unordered(bt_fn, boot_weights), total=num_round))

    ratings = np.array(results)

    scaled_ratings = scale_and_offset(
        ratings=ratings,
        models=models,
        scale=scale,
        init_rating=init_rating,
        baseline_model=baseline_model,
        baseline_rating=baseline_rating,
    )

    df = pd.DataFrame(scaled_ratings, columns=models)
    return df[df.median().sort_values(ascending=False).index]


DIFF_MASK = np.array(
    [1.0, -1.0], dtype=np.float64
)  # create globally to not incur the instantiation cost in each call


def contextual_bt_loss_and_grad(
    params,
    n_competitors,
    matchups,
    features,
    outcomes,
    alpha=1.0,
    reg=1.0,
    half_reg=0.5,
):
    reg_loss = half_reg * np.inner(params, params)

    # Split params into ratings and feature parameters
    ratings = params[:n_competitors]
    feature_params = params[n_competitors:]

    matchup_ratings = ratings[matchups]
    bt_logits = alpha * (matchup_ratings[:, 0] - matchup_ratings[:, 1])
    context_logits = np.dot(features, feature_params)
    probs = expit(bt_logits + context_logits)
    loss = (-((np.log(probs) * outcomes + np.log(1.0 - probs) *
               (1.0 - outcomes))).sum() + reg_loss)

    error = outcomes - probs
    grad = reg * params  # initialize the grad as the regularization grad
    matchups_grads = -alpha * error
    np.add.at(grad[:n_competitors], matchups[:, [0, 1]],
              matchups_grads[:, None] * DIFF_MASK)
    grad[n_competitors:] -= np.dot(features.T, error)
    return loss, grad


# note on regularization:
# default reg is to 0.5 since the LogisticRegression default is 1.0
# in the original implementation, matchups were duplicated
# that made the ratio of log loss to reg loss "twice as high"
# in this non-duplicated version for parity we also reduce the reg by one half to match
def fit_contextual_bt(
        matchups,
        features,
        outcomes,
        models,
        idxs=None,
        alpha=math.log(10.0),
        reg=0.5,
        tol=1e-6,
):
    n_features = features.shape[1]
    n_models = len(models)
    initial_params = np.zeros(n_models + n_features, dtype=np.float64)
    half_reg = reg / 2.0

    # sample idxs optionally allow for fitting on a bootstrap sample of the dataset
    if idxs is not None:
        matchups, features, outcomes = matchups[idxs], features[
            idxs], outcomes[idxs]

    result = minimize(
        fun=contextual_bt_loss_and_grad,
        x0=initial_params,
        args=(n_models, matchups, features, outcomes, alpha, reg, half_reg),
        jac=True,
        method='L-BFGS-B',
        options={
            'disp': False,
            'maxiter': 100,
            'gtol': tol
        },
    )
    return result['x']


def compute_style_control(
    df: pd.DataFrame,
    alpha: float = math.log(10.0),
    reg: float = 0.5,
    scale: float = 400.0,
    init_rating: float = 1000.0,
    baseline_model: str = None,
    baseline_rating: float = 1000.0,
    normalize_style_features: bool = True,
    control_variables: List[str] = None,
    odds_ratio: bool = True,
    tol: float = 1e-6,
):
    if control_variables is not None:
        _df = pd.get_dummies(
            data=df,
            columns=control_variables,
            drop_first=
            False,  # Since the model is fitted without an intercept, we keep all levels of each categorical
        )

        # One-hot encode categorical control variables
        one_hot_ctrls = []
        for col in _df.columns:
            for ctrl_var in control_variables:
                if col.startswith(ctrl_var):
                    one_hot_ctrls.append(col)
                    break

    matchups, features, outcomes, models = preprocess_for_style(
        _df,
        normalize_style_features=normalize_style_features,
        style_variables=STYLE_CONTROL_VARIABLES_V1,
        control_variables=one_hot_ctrls,
    )
    ratings_params = fit_contextual_bt(
        matchups,
        features,
        outcomes,
        models=models,
        alpha=alpha,
        reg=reg,
        tol=tol,
    )
    ratings = ratings_params[:len(models)]

    if odds_ratio:
        params = np.exp(ratings_params[len(models):])
    else:
        params = ratings_params[len(models):]

    scaled_ratings = scale_and_offset(
        ratings=ratings,
        models=models,
        scale=scale,
        init_rating=init_rating,
        baseline_model=baseline_model,
        baseline_rating=baseline_rating,
    )
    scaled_ratings = pd.Series(scaled_ratings,
                               index=models).sort_values(ascending=False)

    control_coefficients = {
        k: v
        for k, v in zip(STYLE_CONTROL_VARIABLES_V1 + one_hot_ctrls, params)
    }

    return scaled_ratings, control_coefficients


def compute_bootstrap_style_control(
    df,
    num_round: int,
    alpha: float = math.log(10.0),
    reg: float = 0.5,
    scale: float = 400.0,
    init_rating: float = 1000.0,
    baseline_model: str = None,
    baseline_rating: float = 1000.0,
    normalize_style_features: bool = True,
    control_variables: List[str] = None,
    odds_ratio: bool = True,
    tol: float = 1e-6,
    num_cpu: int = None,
):
    if control_variables is not None:
        _df = pd.get_dummies(
            data=df,
            columns=control_variables,
            drop_first=
            False,  # Since the model is fitted without an intercept, we keep all levels of each categorical
        )

        # One-hot encode categorical control variables
        one_hot_ctrls = []
        for col in _df.columns:
            for ctrl_var in control_variables:
                if col.startswith(ctrl_var):
                    one_hot_ctrls.append(col)
                    break

    matchups, features, outcomes, models = preprocess_for_style(
        _df,
        normalize_style_features=normalize_style_features,
        style_variables=STYLE_CONTROL_VARIABLES_V1,
        control_variables=one_hot_ctrls,
    )

    contextual_bt_fn = partial(
        fit_contextual_bt,
        matchups,
        features,
        outcomes,
        models,
        alpha=alpha,
        reg=reg,
        tol=tol,
    )

    boot_idxs = np.random.randint(low=0,
                                  high=matchups.shape[0],
                                  size=(num_round, matchups.shape[0]))

    with mp.Pool(num_cpu if num_cpu else os.cpu_count()) as pool:
        results = list(
            tqdm(pool.imap_unordered(contextual_bt_fn, boot_idxs),
                 total=num_round))

    ratings_params = np.array(results)
    ratings = ratings_params[:, :len(models)]

    if odds_ratio:
        params = np.exp(ratings_params[:, len(models):].mean(axis=0))
    else:
        params = ratings_params[:, len(models):].mean(axis=0)

    scaled_ratings = scale_and_offset(
        ratings=ratings,
        models=models,
        scale=scale,
        init_rating=init_rating,
        baseline_model=baseline_model,
        baseline_rating=baseline_rating,
    )
    df = pd.DataFrame(scaled_ratings, columns=models)

    control_coefficients = {
        k: v
        for k, v in zip(STYLE_CONTROL_VARIABLES_V1 + one_hot_ctrls, params)
    }

    return df[df.median().sort_values(
        ascending=False).index], control_coefficients


class CompassArenaBradleyTerrySummarizer(DefaultSubjectiveSummarizer):
    """Summarizer for fitting and Bradley-Terry model to pairwise matchups
    according to https://github.com/lm-sys/FastChat/tree/main.

    Args:
        config (ConfigDict): The configuration object of the evaluation task. It's expected to be filled out at runtime.
        dataset_abbrs (Optional[List[str]], optional): Dataset abbreviations to be listed in the summary. Defaults to None.
        summary_groups (List, optional): Passed to DefaultSubjectiveSummarizer. Not used for this class. Defaults to None.
        prompt_db (_type_, optional): Legacy parameter kept for backward compatibility. Defaults to None.
        rating_system (str, optional): Rating system used. Currently only supports "bradleyterry". Defaults to "bradleyterry".
        report_pred_win_rates (bool, optional): Whether to report the predicted win rates (against the baseline model) instead of the arena ratings. Defaults to True.
        num_bootstrap (int, optional): The number of bootstraps for estimating the confidence intervals. Defaults to 300.
        num_cpu (int, optional): The number of CPUs to use for the BT bootstrapping process. Defaults to None.
        with_control_vars (bool, optional): Whether to include additional covariates (including style features and group variables) when fitting the BT model. Defaults to True.
        normalize_style_features (bool, optional): Whether to normalize style features BEFORE fitting the BT model (implementation by FastChat). Turn this off for easier interpretation of odds ratios (when odds_ratio==True). Defaults to True.
        odds_ratio (bool, optional): Whether to report odds ratios (np.exp(beta_k)) instead of the original coefficients. Defaults to True.
        groups (List[str], optional): Group variables to include while fitting the BT model. These must be available in the input dataset for each observation. Defaults to None.
    """

    def __init__(
        self,
        config: ConfigDict,
        dataset_abbrs: Optional[List[str]] = None,
        summary_groups: List = None,
        prompt_db=None,
        rating_system: str = 'bradleyterry',
        report_pred_win_rates: bool = True,
        num_bootstrap: int = 300,
        num_cpu: int = None,
        with_control_vars: bool = True,
        normalize_style_features: bool = True,
        odds_ratio: bool = True,
        groups: List[str] = None,
    ) -> None:
        summary_groups = [] if summary_groups is None else summary_groups
        super().__init__(config, dataset_abbrs, summary_groups, prompt_db)

        self.summarizer_cfg = self.cfg['summarizer']
        self.rating_system = 'bradleyterry'  # Only bradleyterry supported
        self.report_pred_win_rates = report_pred_win_rates
        self.num_bootstrap = num_bootstrap
        self.num_cpu = num_cpu
        self.with_control_vars = with_control_vars
        self.normalize_style_features = normalize_style_features
        self.odds_ratio = odds_ratio
        self.groups = [] if groups is None else groups

    def _pick_up_results(self, judge_abbr):
        """The function reads the numerical results of evaluations from the
        output folder based on the configuration file, and ultimately returns
        four dictionaries, each containing processed information in different
        formats. The contents of the four dictionaries are as follows:

        - raw_results: contains the raw results of each model on each dataset (excluding details).
        - parsed_results: contains the results of each model on each dataset for each metric, with metrics in METRIC_BLACKLIST being ignored.
        - dataset_metrics: contains the list of metrics for each dataset, consistent with the metrics in parsed_results. The list is ordered according to the METRIC_WHITELIST,
            with metrics appearing earlier considered more important.
        - dataset_eval_mode: contains the evaluation mode for each dataset.
        """
        # raw_results: {model_abbr: {dataset_abbr: result}}
        raw_results: Dict[str, Dict[str, Any]] = {}
        # # parsed_results: {model_abbr: {dataset_abbr: {metric: score}}}
        # parsed_results: Dict[str, Dict[str, Dict[str, float]]] = {}
        # # dataset_metrics: {dataset_abbr: [metric]}
        # dataset_metrics: Dict[str, List[str]] = {}

        for model in self.model_cfgs:
            model_abbr = model_abbr_from_cfg_used_in_summarizer(model)
            # parsed_results.setdefault(model_abbr, {})
            # raw_results.setdefault(model_abbr, {})

            for dataset in self.dataset_cfgs:
                base_models = dataset.get('base_models', None)
                if base_models is None:
                    raise ValueError(
                        'CompassArenaBradleyTerrySummarizer requires at least one `base_model` in specified in the dataset config.'
                    )

                base_models_list = [item['abbr'] for item in base_models]

                dataset_abbr = dataset_abbr_from_cfg(dataset)
                raw_results.setdefault(dataset_abbr, {})

                for base_model_abbr in base_models_list:
                    raw_results[dataset_abbr].setdefault(base_model_abbr, [])

                    origin_path = get_infer_output_path(
                        model, dataset, osp.join(self.work_dir, 'results'))
                    if base_model_abbr != '':
                        temp_path, dataset_json_name = (
                            origin_path.rsplit('/', 1)[0],
                            origin_path.rsplit('/', 1)[1],
                        )
                        filepath = osp.join(
                            temp_path.rsplit('/', 1)[0],
                            base_model_abbr + '_' +
                            temp_path.rsplit('/', 1)[1] + '_judged-by--' +
                            judge_abbr,
                            dataset_json_name,
                        )
                    else:
                        filepath = osp.join(
                            origin_path.rsplit('/', 1)[0] + '_judged-by--' +
                            judge_abbr,
                            origin_path.rsplit('/', 1)[1],
                        )
                    if not osp.exists(filepath):
                        continue

                    result = mmengine.load(filepath)
                    result.pop('details', None)

                    # raw_results[dataset_abbr] = result
                    raw_results[dataset_abbr][base_model_abbr].extend(
                        result['matches'])

                    if 'error' in result:
                        self.logger.debug(
                            f'error in {model_abbr} {dataset_abbr} {result["error"]}'
                        )
                        continue

        # dataset_eval_mode: {dataset_abbr: eval_mode}
        dataset_eval_mode: Dict[str, str] = {}
        for dataset in self.dataset_cfgs:
            inferencer = (dataset.get('infer_cfg', {}).get('inferencer',
                                                           {}).get('type', ''))
            inferencer = (inferencer if isinstance(inferencer, str) else
                          inferencer.__name__)
            dataset_abbr = dataset_abbr_from_cfg(dataset)
            if 'GenInferencer' in inferencer:
                dataset_eval_mode[dataset_abbr] = 'gen'
            elif 'PPLInferencer' in inferencer:
                dataset_eval_mode[dataset_abbr] = 'ppl'
            elif 'LLInferencer' in inferencer:
                dataset_eval_mode[dataset_abbr] = 'll'
            else:
                dataset_eval_mode[dataset_abbr] = 'unknown'
                self.logger.warning(
                    f'unknown inferencer: {inferencer} - {dataset_abbr}')

        # return raw_results, parsed_results, dataset_metrics, dataset_eval_mode
        return raw_results, dataset_eval_mode

    def _calculate_ratings(
        self,
        matches: Dict,
        base_model: str = None,
        groups: List[str] = None,
    ) -> Tuple[pd.DataFrame, Dict]:

        rating_system = self.rating_system
        num_bootstrap = self.num_bootstrap
        num_cpu = self.num_cpu
        with_control_vars = self.with_control_vars

        matches_df = pd.DataFrame(matches)

        num_battles = (matches_df['model_a'].value_counts().add(
            matches_df['model_b'].value_counts(), fill_value=0))

        # if rating_system == "bradleyterry":
        if with_control_vars:
            elo_rating_final, coef_final = compute_style_control(
                df=matches_df,
                baseline_model=base_model,
                normalize_style_features=self.normalize_style_features,
                control_variables=groups,
                odds_ratio=self.odds_ratio,
            )

            bootstrap_df, bootstrap_coef = compute_bootstrap_style_control(
                df=matches_df,
                num_round=num_bootstrap,
                baseline_model=base_model,
                normalize_style_features=self.normalize_style_features,
                control_variables=groups,
                odds_ratio=self.odds_ratio,
            )
        else:
            bootstrap_df = compute_bootstrap_bt(
                battles=matches_df,
                num_round=num_bootstrap,
                baseline_model=base_model,
                num_cpu=num_cpu,
            )
            elo_rating_final = compute_bt(
                df=matches_df,
                baseline_model=base_model,
            )

        # print(elo_rating_final)

        # elif rating_system == "elo":
        #     bootstrap_df = compute_bootstrap_elo(
        #         df=matches_df,
        #         num_round=num_bootstrap,
        #         num_cpu=num_cpu,
        #     )
        #     elo_rating_final = compute_elo(matches_df)

        model_rating_q025 = bootstrap_df.quantile(0.025)
        model_rating_q975 = bootstrap_df.quantile(0.975)

        # compute ranking based on CI
        model_order = list(elo_rating_final.index)

        ranking = {}
        for i, model_a in enumerate(model_order):
            ranking[model_a] = 1
            for j, model_b in enumerate(model_order):
                if i == j:
                    continue
                if model_rating_q025[model_b] > model_rating_q975[model_a]:
                    ranking[model_a] += 1

        leaderboard_table_df = pd.DataFrame(
            {
                'rating': elo_rating_final,
                'ranking_ub': pd.Series(ranking),
                'std_dev': bootstrap_df.std(),
                'rating_q975': model_rating_q975,
                'rating_q025': model_rating_q025,
                'num_battles': num_battles,
            }, )
        leaderboard_table_df['model_name'] = leaderboard_table_df.index

        leaderboard_table_df.sort_values(
            by=['rating'],
            ascending=False,
            inplace=True,
        )
        leaderboard_table_df['ranking'] = np.arange(
            1,
            len(leaderboard_table_df) + 1)

        if rating_system == 'bradleyterry' and with_control_vars:
            control_coefficients = {
                'bootstrap': bootstrap_coef,
                'final': coef_final,
            }
        else:
            control_coefficients = {'final': []}

        return leaderboard_table_df, control_coefficients['final']

    def _output_to_file(
        self,
        output_path,
        time_str: str,
        tables: Dict,
        metadata: Dict,
        judge_abbr: str,
        dataset_eval_mode: str,
    ):
        # Output to file
        if output_path is None:
            output_path = osp.join(self.work_dir, 'summary',
                                   f'summary_{time_str}.json')
            output_csv_path = osp.join(self.work_dir, 'summary',
                                       f'summary_{time_str}.csv')
        else:
            output_csv_path = output_path.replace('.json', '.csv')
        output_path = output_path.split(
            '.json')[0] + '_by_' + judge_abbr + '.json'

        output_dir = osp.split(output_path)[0]
        mmengine.mkdir_or_exist(output_dir)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        self.logger.info(f'write summary to {osp.abspath(output_path)}')

        prompt_version = {
            dataset_abbr_from_cfg(d): get_prompt_hash(d)[:6]
            for d in self.dataset_cfgs
        }

        full_results = []
        for base_model_abbr, datasets in tables.items():
            base_model_results = []
            for dataset_abbr, table_df in datasets.items():
                table_df['dataset'] = dataset_abbr
                table_df['version'] = prompt_version.get(dataset_abbr, '-')
                table_df['metric'] = 'bt_rating'
                table_df['mode'] = dataset_eval_mode[dataset_abbr]
                table_df['base_model'] = base_model_abbr

                base_model_results.append(table_df)

            cur_base_model_result_df = pd.concat(base_model_results)
            full_results.append(cur_base_model_result_df)

        full_results_df = pd.concat(full_results)
        full_results_df = full_results_df[[
            'dataset',
            'version',
            'base_model',
            'metric',
            'mode',
            'ranking',
            'ranking_ub',
            'model_name',
            'predicted_win_rate',
            'rating',
            'rating_q975',
            'rating_q025',
            'std_dev',
            'num_battles',
        ]]

        output_csv_path = (output_csv_path.split('.csv')[0] + '_by_' +
                           judge_abbr + '.csv')

        with pd.option_context(
                'display.max_rows',
                20,
                'display.max_columns',
                20,
                'display.expand_frame_repr',
                False,
        ):
            print(full_results_df.reset_index(drop=True).round(2))

        full_results_df.to_csv(
            output_csv_path,
            index=False,
        )
        self.logger.info(f'write csv to {osp.abspath(output_csv_path)}')

    def flip_dict_levels(self, original_dict: Dict):
        """Flips the two levels of a nested dictionary so that dict[lvl1][lvl2]
        becomes dict[lvl2][lvl1].

        Args:
            original_dict (dict): The original nested dictionary.

        Returns:
            dict: The flipped dictionary.
        """
        flipped_dict = {}
        for lvl1, lvl2_dict in original_dict.items():
            for lvl2, value in lvl2_dict.items():
                if lvl2 not in flipped_dict:
                    flipped_dict[lvl2] = {}
                flipped_dict[lvl2][lvl1] = value

        return flipped_dict

    def predict_win_rate(
        self,
        ratings_df: pd.DataFrame,
        baseline_model: str,
        base: float = 10.0,
        scaling_factor: float = 400.0,
        round_win_rate: int = None,
    ) -> pd.DataFrame:
        """Predict win rates between all models using their ELO ratings.

        Args:
            ratings_df (pd.DataFrame): DataFrame containing model ratings with model names as index
            baseline_model (str): Name of baseline model to use as reference
            base (float): Base for the ELO formula (default 10.0)
            scaling_factor (float): Scaling factor for rating differences (default 400.0)

        Returns:
            pd.DataFrame: DataFrame with an additional column 'predicted_win_rate' containing
                the predicted win rate against the baseline model
        """
        if baseline_model not in ratings_df.index:
            raise ValueError(
                f'Baseline model {baseline_model} not found in ratings')

        # Create a copy of the ratings dataframe to avoid modifying the original
        result_df = ratings_df.copy()

        # Initialize the predicted_win_rate column with 0.5 for the baseline model

        result_df['predicted_win_rate'] = 0.5

        # Get the baseline model's rating
        baseline_rating = ratings_df.loc[baseline_model, 'rating']

        # Calculate win probabilities for all models against the baseline
        for model, row in ratings_df.iterrows():
            if model != baseline_model:
                model_rating = row['rating']
                # ELO win probability formula
                win_rate = 1 / (1 + base**(
                    (baseline_rating - model_rating) / scaling_factor))
                result_df.loc[model, 'predicted_win_rate'] = win_rate

        if round_win_rate is not None:
            result_df['predicted_win_rate'] = result_df[
                'predicted_win_rate'].round(round_win_rate)

        return result_df

    def summarize(
            self,
            output_path: str = None,
            time_str: str = datetime.now().strftime('%Y%m%d_%H%M%S'),
    ):
        """Summarize evaluation results and format output table.

        Args:
            output_path (str, optional): Output path. Defaults to None.
            time_str (str, optional): Timestamp for file suffix. Defaults to
            datetime.now().strftime('%Y%m%d_%H%M%S').
        """
        all_scores_df_list = []
        all_scores = {}
        all_scores_ctrl_coefs = {}
        for judge_model in self.judge_models:
            control_coefficients = {}
            leaderboard_tables = {}

            judge_abbr = model_abbr_from_cfg(judge_model)

            # pick up results
            raw_results, dataset_eval_mode = self._pick_up_results(judge_abbr)

            all_matches = []
            for dataset_abbr, base_models in raw_results.items():
                control_coefficients[dataset_abbr] = {}
                leaderboard_tables[dataset_abbr] = {}

                dataset_matches = base_models[list(base_models)[0]]
                all_matches.extend(dataset_matches)

                for base_model_abbr, matches in base_models.items():
                    cur_table_df, cur_ctrl_coefs = self._calculate_ratings(
                        matches=matches,
                        base_model=base_model_abbr,
                        groups=self.groups,
                    )

                    # Calculate predicted win_rate
                    cur_table_df = self.predict_win_rate(
                        ratings_df=cur_table_df,
                        baseline_model=base_model_abbr,
                        round_win_rate=4,
                    )

                    control_coefficients[dataset_abbr][
                        base_model_abbr] = cur_ctrl_coefs
                    leaderboard_tables[dataset_abbr][
                        base_model_abbr] = cur_table_df

                    print('-' * 10 +
                          f"{dataset_abbr + ':' + base_model_abbr}\n" +
                          '-' * 10)
                    print(cur_table_df)
                    print(cur_ctrl_coefs)

            leaderboard_tables = self.flip_dict_levels(leaderboard_tables)

            # Output to .json / .csv files
            self._output_to_file(
                output_path=output_path,
                time_str=time_str,
                tables=leaderboard_tables,
                metadata=control_coefficients,
                judge_abbr=judge_abbr,
                dataset_eval_mode=dataset_eval_mode,
            )

            # Fit another BT model with the first base_model and combining matches from all datasets
            cur_judge_all_scores_df, cur_judge_all_scores_ctrl_coefs = (
                self._calculate_ratings(
                    matches=all_matches,
                    base_model=list(base_models)[0],
                    groups=self.groups,
                ))
            # Calculate predicted win_rate
            cur_judge_all_scores_df = self.predict_win_rate(
                ratings_df=cur_judge_all_scores_df,
                baseline_model=list(base_models)[0],
                round_win_rate=4,
            )
            cur_judge_all_scores_df['judge'] = judge_abbr

            all_scores_df_list.append(cur_judge_all_scores_df)

            # Report predicted win rate or ratings
            if self.report_pred_win_rates:
                _scores = cur_judge_all_scores_df['predicted_win_rate']
            else:
                _scores = cur_judge_all_scores_df['rating']

            all_scores[judge_abbr] = pd.Series(
                _scores,
                index=cur_judge_all_scores_df['model_name'],
            ).to_dict()

            all_scores_ctrl_coefs[judge_abbr] = cur_judge_all_scores_ctrl_coefs

        all_scores_df = pd.concat(all_scores_df_list)

        output_path_all_scores_df = osp.join(
            self.work_dir, 'summary', f'summary_{time_str}_all_scores_df.csv')
        output_path_all_scores = osp.join(
            self.work_dir, 'summary', f'summary_{time_str}_all_scores.json')
        output_path_all_scores_ctrl_coefs = osp.join(
            self.work_dir, 'summary',
            f'summary_{time_str}_all_scores_ctrl_coefs.json')

        all_scores_df.to_csv(output_path_all_scores_df)

        with open(output_path_all_scores, 'w', encoding='utf-8') as f:
            json.dump(all_scores, f, ensure_ascii=False, indent=4)

        with open(output_path_all_scores_ctrl_coefs, 'w',
                  encoding='utf-8') as f:
            json.dump(all_scores_ctrl_coefs, f, ensure_ascii=False, indent=4)

        print(f'{all_scores_df=}')
        print(f'{all_scores=}')
        print(f'{all_scores_ctrl_coefs=}')

        return {'CompassArenaSubjBenchBradleyTerry': all_scores}
