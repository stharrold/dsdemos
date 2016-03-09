#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Pytests for dsdemos/ml.py

"""


# Import standard packages.
import os
import sys
# Import installed packages.
# Import local packages.
sys.path.insert(0, os.path.curdir)
import dsdemos.ml as ml


# TODO: def test__unweight_target
# TODO: def test_MapUniqueToTargetMedian
#     Test with/without weights
#     Test docstring example
#     Test raises
# TODO: def test_MapTargetToNormalDist
#     Test with method=all
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
