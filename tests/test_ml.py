#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Pytests for dsdemos/ml.py

"""


# Import standard packages.
import collections
import os
import sys
import warnings
# Import installed packages.
import numpy as np
import pandas as pd
# Import local packages.
sys.path.insert(0, os.path.curdir)
import dsdemos.ml as ml


# TODO: def test__unweight_target
# TODO: def test_StepTargetToNormalDist
#     Test with method=all
#     Test with/without weights
#     Test docstring example
#     Test raises
# TODO: def test_StepRemoveConstantFeatures
#     Test with/without weights
#     Test docstring example
#     Test raises

def test_StepReplaceCategoricalWithTarget(
    df_features:pd.DataFrame=pd.DataFrame(
        data=[[10, 'a'], [10, 'b'], [10, 'a'], [11, 'b'], [11, 'a'], [11, 'b']],
        columns=['ftr0', 'ftr1']),
    ds_target:pd.Series=pd.Series(data=[100, 101, 102, 103, 104, 105]),
    ds_weight:pd.Series=pd.Series(data=[1, 1, 1, 1, 1, 1]),
    cat_features:list=['ftr0', 'ftr1'],
    features_orig_mean:collections.defaultdict=None,
    warn:bool=False,
    ref_features_orig_mean:collections.defaultdict=collections.defaultdict(
        lambda: 102.5,
        {'ftr0': collections.defaultdict(lambda: 102.5, {10: 101, 11: 104}),
         'ftr1': collections.defaultdict(lambda: 102.5, {'a': 102, 'b': 103})}),
    ref_features_mean_orig:collections.defaultdict=collections.defaultdict(
        lambda: None,
        {'ftr0': collections.defaultdict(lambda: None, {101: 10, 104: 11}),
         'ftr1': collections.defaultdict(lambda: None, {102: 'a', 103: 'b'})}),
    ref_df_features_tform:pd.DataFrame=pd.DataFrame(
        data=[[101, 102], [101, 103], [101, 102], [104, 103], [104, 102], [104, 103]],
        columns=['ftr0', 'ftr1']),
    ref_ds_target_tform:pd.Series=pd.Series(data=[100, 101, 102, 103, 104, 105]),
    ref_ds_weight_tform:pd.Series=pd.Series(data=[1, 1, 1, 1, 1, 1])) -> None:
    r"""Pytest for StepReplaceCategoricalWithTarget.
    
    """
    # Test `__init__`: Test assigned instance attributes.
    cls = ml.StepReplaceCategoricalWithTarget(
        cat_features=cat_features,
        features_orig_mean=features_orig_mean,
        warn=warn)
    assert cat_features == cls.cat_features
    assert features_orig_mean == cls.features_orig_mean
    if cls.features_orig_mean is not None:
        assert ref_features_mean_orig == cls.features_mean_orig
    # Test `fit`: Test assigned as instance attributes.
    cls.fit(
        df_features=df_features,
        ds_target=ds_target,
        ds_weight=ds_weight,
        warn=warn)
    assert ref_features_orig_mean == cls.features_orig_mean
    assert ref_features_mean_orig == cls.features_mean_orig
    # Test `transform`: Test returned data frames/series.
    (test_df_features_tform, test_ds_target_tform, test_ds_weight_tform) = \
        cls.transform(
            df_features=df_features,
            ds_target=ds_target,
            ds_weight=ds_weight)
    assert np.all(ref_df_features_tform == test_df_features_tform)
    assert np.all(ref_ds_target_tform == test_ds_target_tform)
    assert np.all(ref_ds_weight_tform == test_ds_weight_tform)
    # Test `inverse_transform`: Test returned data frames/series match
    #     untransformed originals.
    # Note: Does not pass for ill-defined inverse feature-orig-mean mappings.
    (test_df_features_itform, test_ds_target_itform, test_ds_weight_itform) = \
        cls.inverse_transform(
            df_features=test_df_features_tform,
            ds_target=test_ds_target_tform,
            ds_weight=test_ds_weight_tform)
    assert np.all(df_features == test_df_features_itform)
    assert np.all(ds_target == test_ds_target_itform)
    assert np.all(ds_weight == test_ds_weight_itform)
    return None


# TODO: def test_StepReplaceCategoricalWithTarget_suppl():
#     * Supplemental pytests for StepReplaceCategoricalWithTarget
#     * Test `_calc_features_mean_orig`: Test as an instance method.
#         cls._calc_features_mean_orig(default_factory=None, warn=warn)
#         assert ref_features_mean_orig == cls.features_mean_orig
#     * assert that `_invert_defaultdict` raises RuntimeWarning
#         when mappings are not unique.
#     * assert that `__init__` raises ValueError if some features in
#         `cat_features` are not in `features_orig_mean`.
#     * assert that `fit` raises:
#         * ValueError if some features in `cat_features` are not in
#             `features_orig_mean`.
#         * RuntimeWarning when mappings are not unique.
#     * assert that `transform` raises ValueError
#         * if `self.features_orig_mean`
#         * if some features in `cat_features` are not in `features_orig_mean`.
#     * assert that `inverse_transform` raises ValueError
#         *  if `self.features_mean_orig` is undefined,
#             i.e. if `features_orig_mean` was not assigned with the class
#             initialization or if `fit` has not already been called.
#         * if some features in `self.features_orig_mean` are not
#             in `df_features.columns`.
# TODO: def test_StepFeaturesToRobustScale
#     Test with/without weights
#     Test docstring example
#     Test raises
# TODO: def test_calc_feature_importances
#     Use iris data set, set random seed to 0
#     Test with/without weights
# TODO: def test_calc_score_pvalue
#     Test with/without weights
#     Compare to
#     http://scikit-learn.org/stable/auto_examples/feature_selection/
#     plot_permutation_test_for_classification.html
# TODO: def test_plot_silhouette_scores
#     compare to http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
# TODO: def test_plot_actual_vs_predicted
