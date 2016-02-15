#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Utilities for machine learning.

"""


# Import standard packages.
import collections
import warnings
# Import installed packages.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# Import local packages.
import dsdemos.utils as utils


def calc_feature_importances(
    estimator,
    df_features:pd.DataFrame, ds_target:pd.Series, ds_weight:pd.Series=None, 
    size_sub:int=None, replace:bool=True, dists:list=None, 
    show_progress:bool=False, show_plot:bool=False) -> pd.DataFrame:
    r"""Calculate feature importances and compare to random added features.
    
    Parameters:
        estimator: Estimator model from `sklearn` with attributes `fit`
            and, after running `fit`, also `feature_importances_`.
            See 'Raises', 'Notes'.
        df_features (pandas.DataFrame): Data frame of feature values.
            Format: rows=records, cols=features.
        ds_target (pandas.Series): Data series of target values.
            Format: rows=records, col=target.
        ds_weight (pandas.Series, optional, None): Data series of record
            weights. Format: rows=records, col=weight.
            Default: `None`, then all set to 1.
        size_sub (int, optional, None): Number of records in subset for
            selecting features. Default: `None`, then all records are used.
            The larger `size_sub` the more features that will become
            significant. See 'Raises'.
        replace (bool, optional, True): Sample records with replacement
            (bootstrap sampling). Default: `True`.
            Use `replace=True` to reduce overfitting the feature importances.
        dists (list, optional, None): List of random distributions to add into
            `df_features` to compare significance of feature importance.
            List items are evaluated with `eval` then appended to `df_features`.
            Default: `None`, then uses distributions Laplace, logistic,
            lognormal, Pareto, Poisson, Rayleigh, standard Cauchy,
            standard exponential, standard normal, uniform.
        show_progress (bool, optional, False): Whether or not to print status.
        show_plot (bool, optional, False): Whether or not to show summary plot
            of most significant feature importances.
        
    Returns:
        df_importances (pandas.DataFrame): Data frame of feature importances.
            Format: rows=iterations, cols=features+'random'.
          
    Raises:
        ValueError:
            * Raised if not `hasattr(estimator, 'fit')`
        RuntimeWarning:
            * Raised if not `size_sub <= len(df_features)`.
    
    Notes:
        * Feature importance is the normalized reduction in the loss score.
            See the `sklearn` documentation for your estimator and
            the estimator's 'feature_importances_' attribute.
    
    """
    # TODO: Replace show_progress and warnings.warn with logger.[debug,warn]
    #     https://github.com/stharrold/stharrold.github.io/issues/58
    # Check arguments.
    # Note: Copy df_features to avoid modifying input data.
    if not hasattr(estimator, 'fit'):
        raise ValueError(
            ("`estimator` must have the attribute 'fit'.\n" +
             "Required: hasattr(estimator, 'fit')"))
    if (size_sub is not None) and not (size_sub <= len(df_features)):
        warnings.warn(
            ("The number of records in the subset for calculating feature\n" +
             "importances is larger than the number of records in the data.\n" +
             "Suggested: size_sub <= len(df_features)\n" +
             "Given: {lhs} <= {rhs}").format(
                 lhs=size_sub, rhs=len(df_features)),
             RuntimeWarning)
    df_ftrs_rnd = df_features.copy()
    size_data = len(df_features)
    if size_sub is None:
        size_sub = size_data
    dists = [
        'np.random.laplace(loc=0.0, scale=1.0, size=size_data)',
        'np.random.logistic(loc=0.0, scale=1.0, size=size_data)',
        'np.random.lognormal(mean=0.0, sigma=1.0, size=size_data)',
        'np.random.pareto(a=1.0, size=size_data)',
        'np.random.poisson(lam=1.0, size=size_data)',
        'np.random.rayleigh(scale=1.0, size=size_data)',
        'np.random.standard_cauchy(size=size_data)',
        'np.random.standard_exponential(size=size_data)',
        'np.random.standard_normal(size=size_data)',
        'np.random.uniform(low=-1.0, high=1.0, size=size_data)']
    # Include different of randomized features and evaluate their importance,
    # one at a time.
    ftrs_imps = collections.defaultdict(list)
    if show_progress:
        print("Progress:", end=' ')
    for (inum, dist) in enumerate(dists):
        df_ftrs_rnd['random'] = eval(dist)
        idxs_sub = np.random.choice(a=size_data, size=size_sub, replace=replace)
        if ds_weight is None:
            estimator.fit(
                X=df_ftrs_rnd.values[idxs_sub], y=ds_target.values[idxs_sub])            
        else:
            estimator.fit(
                X=df_ftrs_rnd.values[idxs_sub], y=ds_target.values[idxs_sub],
                sample_weight=ds_weight.values[idxs_sub])
        for (col, imp) in zip(
            df_ftrs_rnd.columns, estimator.feature_importances_):
            ftrs_imps[col].append(imp)
        if show_progress:
            print("{frac:.0%}".format(frac=(inum+1)/len(dists)), end=' ')
    if show_progress:
        print('\n')
    # Return the feature importances and plot most important features.
    # Plotted features have (average importance) > max(random importance)
    df_importances = pd.DataFrame.from_dict(ftrs_imps)
    if show_plot:
        ds_ftrs_imps_mean = df_importances.mean()
        tfmask = (ds_ftrs_imps_mean > df_importances['random'].max())
        ds_ftrs_top = ds_ftrs_imps_mean[tfmask].sort_values(ascending=False)
        idxs_ftrs_plot = ds_ftrs_top.index.append(pd.Index(['random']))
        sns.barplot(
            data=df_importances[idxs_ftrs_plot], order=idxs_ftrs_plot, ci=95,
            orient='h', color=sns.color_palette()[0])
        plt.title("Feature column name vs importance score")
        plt.xlabel("Importance score\n" +
                   "(normalized reduction of loss function)")
        plt.ylabel("Feature column name")
        plt.show()
    return df_importances
