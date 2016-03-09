#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Utilities for machine learning.

"""


# Import standard packages.
import collections
import itertools
import warnings
# Import installed packages.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sklearn.cross_validation as sk_cv
import sklearn.cluster as sk_cl
import sklearn.metrics as sk_met
import sklearn.preprocessing as sk_pre
# Import local packages.
import dsdemos.utils as utils


def _unweight_target(
    ds_target:pd.Series, ds_weight:pd.Series) -> np.ndarray:
    r"""Pseudo-private method to unweight target values.
    
    Args:
        ds_target (pandas.Series): Data series of target values.
            Format: rows=records, col=target.
        ds_weight (pandas.Series, optional, None): Data series of record
            weights. Format: rows=records, col=weight.

    Returns:
        target_vals (numpy.ndarray): 1D array of unweighted target values with
            `len(target_vals) == ds_weight.sum()`.
    
    Raises:
        ValueError: Raised if `ds_target` and `ds_weight` have different number
            of records.
    
    """
    # Check arguments.
    if not len(ds_target) == len(ds_weight):
        raise ValueError(
            ("`ds_target` and `ds_weight` have different number of records.\n" +
             "Required: len(ds_target) == len(ds_weight)\n" +
             "Given: {lhs} == {rhs}").format(
                 lhs=len(ds_target), rhs=len(ds_weight)))
    # Unweight target values.
    target_vals = list()
    for tup in zip(ds_target.values, ds_weight.astype(int)):
        target_vals += itertools.repeat(*tup)
    target_vals = np.asarray(target_vals)
    # Check returns.
    if not len(target_vals) == ds_weight.sum():
        raise AssertionError(
            ("Program error. Number of target values does not match\n" +
             "sum of weights.\n" +
             "Required: len(target_vals) == ds_weight.sum()\n" +
             "Given: {lhs} == {rhs}").format(
                 lhs=len(target_vals), rhs=ds_weight.sum()))
    return target_vals
    

class MapTargetToNormalDist:
    r"""Pipeline step: Keep only positive target values and transform
    unweighted values to be normally distributed using a Box-Cox power
    transformation.

    Notes:
        * Motivation: A mean-squared-error loss function performs better when
            the target value is normally distributed.
        * Box-Cox transformation:[^wiki]
            y = (x^lam - 1)/lam    , if lam != 0
            y = ln(x)              , if lam = 0
        * Inverse Box-Cox transformation:
            x = (y*lam + 1)^(1/lam), if lam != 0
            x = exp(y)             , if lam = 0
    
    Examples:
        * Example fit, transform, and inverse transform:
            Note: Only records with target values > 0 were kept by the
                transform. The inverse transformation does not replace the
                dropped records.
            ```
            >>> (ftrs, trg, wt) = (df_features, ds_target, ds_weight)
            >>> mapping = MapTargetToNormalDist()
            >>> mapping.fit(ftrs, trg, wt)
            >>> (ftrs_tform, trg_tform, wt_tform) =\
            ... mapping.transform(ftrs, trg, wt)
            >>> (ftrs_itform, trg_itform, wt_itform) =\
            ... mapping.inverse_transform(ftrs_tform, trg_tform, wt_tform)
            >>> # Some records were dropped:
            >>> len(ftrs_itform) <= len(ftrs)
            True
            >>> len(trg_itform) <= len(trg)
            True
            >>> len(wt_itform) <= len(wt)
            True
            ```

    See Also:
        sklearn.pipeline.Pipeline, scipy.stats.boxcox

    References:
        [^wiki]: https://en.wikipedia.org/wiki/Power_transform

    """
    
    
    def __init__(self, lmbda:float=None) -> None:
        r"""Pseudo-private method to instantiate class.
        
        Args:
            self (implicit)
            lmbda (float, optional, None): Optimal transformation
                parameter, e.g. found by `scipy.stats.boxcox_normmax`.
                If default (`None`), compute using `fit` method.
            
        Returns:
            None
        
        See Also:
            self.fit
        
        """
        self.lmbda = lmbda
        return None


    def fit(
        self, df_features:pd.DataFrame, ds_target:pd.Series,
        ds_weight:pd.Series=None, **kwargs) -> None:
        r"""Fit the optimal Box-Cox power transformation parameter using
        only positive target values.

        Args:
            self (implicit)
            df_features (pandas.DataFrame): Data frame of feature values.
                Format: rows=records, col=features. See 'Notes'.
            ds_target (pandas.Series): Data series of target values.
                Format: rows=records, col=target.
            ds_weight (pandas.Series, optional, None): Data series of record
                weights. Format: rows=records, col=weight.
            **kwargs (dict): Optional keyword arguments for
                `scipy.stats.boxcox_normmax`
                Example: {brack=(-2.0, 2.0), method='pearsonr'}
            
        Returns:
            None
        
        Attributes:
            lmbda (float): Set by calling `fit`. Optimal transformation
                parameter found by `scipy.stats.boxcox_normmax`. See 'Notes'.
            
        See Also:
            self.transform, scipy.stats.boxcox_normmax
        
        Notes:
            * `df_features` is not used in `fit`. The argument is passed to
                maintain a consistent API for `sklearn.pipeline.Pipeline`.
            * If `kwargs['method']='all'`, then `lmbda` is only assigned
                to the first parameter returned by `scipy.stats.boxcox_normmax`.

        """
        # Check arguments.
        # Copy data frames/series to avoid modifying input data.
        # df_features not used.
        ds_target = ds_target.copy()
        if ds_weight is not None:
            ds_weight = ds_weight.copy()
        # Keep only positive target values.
        tfmask = ds_target > 0
        ds_target = ds_target.loc[tfmask]
        if ds_weight is not None:
            ds_weight = ds_weight.loc[tfmask]
        # Unweight the target values and compute transformation parameter.
        if ds_weight is not None:
            target_vals = _unweight_target(
                ds_target=ds_target, ds_weight=ds_weight)
        else:
            target_vals = ds_target.values
        self.lmbda = scipy.stats.boxcox_normmax(target_vals, **kwargs)
        if hasattr(self.lmbda, '__iter__'):
            self.lmbda = self.lmbda[0]
        return None


    def transform(
        self, df_features:pd.DataFrame, ds_target:pd.Series,
        ds_weight:pd.Series=None, lmbda:float=None) -> tuple:
        r"""Transform data frames/series by keeping only records with positive
        target values and transforming unweighted target values to be normally
        distributed using a Box-Cox power transformation. Suffix '_norm' is
        added to `ds_target.name`.
        
        Args:
            self (implicit)
            df_features (pandas.DataFrame): Data frame of feature values.
                Format: rows=records, col=features.
            ds_target (pandas.Series): Data series of target values.
                Format: rows=records, col=target.
            ds_weight (pandas.Series, optional, None): Data series of record
                weights. Format: rows=records, col=weight.
            lmbda (float, optional, None): Optimal transformation parameter.
            If default (`None`), must be previously set by `fit` method.
            See 'Raises'.
        
        Returns:
            df_features_tform (pandas.DataFrame): Data frame of feature values
                for records with positive target values.
                Format: rows=records, col=features.
            ds_target_tform (pandas.Series): Data series of positive target
                values transformed to be normally distributed when unweighted.
                Format: rows=records, col=target.
            ds_weight_tform (pandas.Series, optional, None): Data series of
                record weights for records with positive target values.
                Format: rows=records, col=weight.
        
        Raises:
            ValueError: Raised if `lmbda` is undefined,
                i.e. if `lmbda is None` and `fit` has not already
                been called.
        
        See Also:
            self.fit, self.inverse_transform
        
        Notes:
            * Box-Cox transformation:[^wiki]
                y = (x^lam - 1)/lam    , if lam != 0
                y = ln(x)              , if lam = 0
        
        """
        # Check arguments.
        if lmbda is not None:
            self.lmbda = lmbda
        if self.lmbda is None:
            raise ValueError(
                "`lmbda` is undefined. Assign `lmbda`\n" +
                "or call `fit`.\n")
        # Copy data frames/series to avoid modifying input data.
        df_features_tform = df_features.copy()
        ds_target_tform = ds_target.copy()
        if ds_weight is not None:
            ds_weight_tform = ds_weight.copy()
        else:
            ds_weight_tform = ds_weight
        # Keep only positive target values.
        tfmask = ds_target_tform > 0
        df_features_tform = df_features_tform.loc[tfmask]
        ds_target_tform = ds_target_tform.loc[tfmask]
        if ds_weight_tform is not None:
            ds_weight_tform = ds_weight_tform.loc[tfmask]
        # Transform the target values.
        ds_target_tform = ds_target_tform.apply(
            lambda x_val: scipy.stats.boxcox(x_val, lmbda=self.lmbda))
        ds_target_tform.name += '_norm'
        return (df_features_tform, ds_target_tform, ds_weight_tform)


    def _inverse_boxcox(self, y_vals, lmbda:float):
        r"""Pseudo-private method to invert `scipy.stats.boxcox`.
        
        Args:
            self (implicit)
            y_vals (float or numpy.ndarray): Box-Cox transformed values to
                inverse transform.
            lmbda (float): Box-Cox transformation parameter.
        
        Returns:
            x_vals (float or numpy.ndarray): Inverse transformed values with
                same shape as `y_vals`.
        
        See Also:
            scipy.stats.boxcox, self.inverse_transform
        
        Notes:
            * Inverse Box-Cox transformation:
                x = (y*lam + 1)^(1/lam), if lam != 0
                x = exp(y)             , if lam = 0
        
        """
        y_vals = np.asarray(y_vals)
        if lmbda == 0:
            x_vals = np.exp(y_vals)
        else:
            x_vals = (y_vals*lmbda + 1)**(1/lmbda)
        return x_vals


    def inverse_transform(
        self, df_features:pd.DataFrame, ds_target:pd.Series,
        ds_weight:pd.Series=None, lmbda:float=None) -> tuple:
        r"""Inverse transform target values from normal distribution to original
        distribution by inverted Box-Cox power transformation. Suffix '_orig' is
        added to `ds_target.name`.
        
        Args:
            self (implicit)
            df_features (pandas.DataFrame): Data frame of feature values
                for records with positive original target values.
                Format: rows=records, col=features. See 'Notes'.
            ds_target (pandas.Series): Data series of positive target values
                normally distributed when unweighted.
                Format: rows=records, col=target. See 'Raises'.
            ds_weight (pandas.Series, optional, None): Data series of
                record weights for records with positive original target values.
                Format: rows=records, col=weight. See 'Notes'.
            lmbda (float, optional, None): Optimal transformation parameter.
                If default (`None`), must be set by calling `fit`.
                See 'Raises'.
        
        Returns:
            df_features_itform (pandas.DataFrame): Data frame of feature values
                for records with positive original target values.
                Format: rows=records, col=features.
            ds_target_itform (pandas.Series): Data series of positive target
                values inverse transformed to be distributed like original
                when unweighted. Format: rows=records, col=target.
            ds_weight_itform (pandas.Series, optional, None): Data series of
                record weights for records with positive original target values.
                Format: rows=records, col=weight.
        
        Raises:
            ValueError:
                * Raised if any target values are <= 0,
                    i.e. if `not numpy.all(ds_target > 0)`.
                * Raised if `lmbda` is undefined,
                    i.e. if `lmbda is None` and `fit` has not already
                    been called.
        
        See Also:
            scipy.stats.boxcox, self.transform
        
        Notes:
            * Inverse Box-Cox transformation:
                x = (y*lam + 1)^(1/lam), if lam != 0
                x = exp(y)             , if lam = 0
            * Only records with target values > 0 were kept by `transform`.
                The `inverse_transform` does not replace the dropped records.
            * `df_features` and `ds_weight` are not used in `inverse_transform`.
                The arguments are passed to maintain a consistent API.

        """
        # Check arguments.
        if not np.all(ds_target > 0):
            raise ValueError(
                "All target values must be > 0.\n" +
                "Required: `numpy.all(ds_target > 0)`")
        if lmbda is not None:
            self.lmbda = lmbda
        if self.lmbda is None:
            raise ValueError(
                "`lmbda` is undefined. Assign `lmbda` or call `fit`.")
        # Copy data frames/series to avoid modifying input data.
        df_features_itform = df_features.copy()
        ds_target_itform = ds_target.copy()
        if ds_weight is not None:
            ds_weight_itform = ds_weight.copy()
        else:
            ds_weight_itform = ds_weight
        # Inverse transform the target values.
        ds_target_itform = ds_target_itform.apply(
            lambda y_val: self._inverse_boxcox(y_val, lmbda=self.lmbda))
        ds_target_itform.name += '_orig'
        return (df_features_itform, ds_target_itform, ds_weight_itform)


class MapUniqueToTargetMedian:
    r"""Pipeline step: Create features from informative priors of
    categorical features.
    
    Notes:
        * Created (transformed/inverse transformed) features are appended as
            columns. No columns are deleted.
        * Motivation: This transformation includes an informative prior into
            the feature, i.e. the feature's group-wise relationship to the
            target. Splits in the transformed feature now correspond to splits
            in the target value.

    Examples:
        * Example fit, transform, and inverse transform:
            A categorical feature, 'cat_ftr',
            is transformed into 'cat_ftr_med',
            then 'cat_ftr_med' is inverse transformed into 'cat_ftr_med_orig'.
        * Created (transformed/inverse transformed) features are appended as
            columns. No columns are deleted.
            ```
            >>> # 'cat_ftr' in `df_features` is a categorical feature
            >>> cat_features = ['ftr0', 'ftr1']
            >>> (ftrs, trg, wt) = (df_features, ds_target, ds_weight)
            >>> mapping = MapUniqueToTargetMedian(cat_features)
            >>> mapping.fit(ftrs, trg, wt)
            >>> (ftrs_tform, trg_tform, wt_tform) =\
            ... mapping.transform(ftrs, trg, wt)
            >>> # 'cat_ftr_med' was added to `ftrs_tform`
            >>> len(ftrs_tform.columns) > len(ftrs.columns)
            True
            >>> (ftrs_itform, trg_itform, wt_itform) =\
            ... mapping.inverse_transform(ftrs_tform, trg_tform, wt_tform)
            >>> # 'cat_ftr_med_orig' was added to `ftrs_itform`
            >>> len(ftrs_itform.columns) > len(ftrs_tform.columns)
            True
            >>> # 'cat_ftr' equals 'cat_ftr_med_orig'
            >>> np.all(ftrs['cat_ftr'] == ftrs_itform['cat_ftr_med_orig'])
            True
            ```

    See Also:
        sklearn.pipeline.Pipeline
    
    """
    
    
    def __init__(
        self, cat_features:list, feature_orig_med:dict=None) -> None:
        r"""Pseudo-private method to instantiate class.
        
        Args:
            self (implicit)
            cat_features (list): List of column names that are categorical
                features. Example: ['ftr0', 'ftr1']
            feature_orig_med (collections.defaultdict, optional, None): Nested
                dict as mapping of categorical feature to original values,
                then from original values to group's unweighted target median.
                Missing keys map to population's unweighted target median.
                If default (`None`), compute using `fit` method.
                Example: {'ftr0': {'a': 0, 'b': 1}, 'ftr1': {'A': 10, 'B': 20}}
    
        Returns:
            None
            
        See Also:
            self.fit
        
        """
        self.cat_features = cat_features
        self.feature_orig_med = feature_orig_med
        return None
    
    
    def fit(
        self, df_features:pd.DataFrame, ds_target:pd.Series,
        ds_weight:pd.Series=None, cat_features:list=None) -> None:
        r"""Fit categorical features to unweighted target medians by group.

        Args:
            self (implicit)
            df_features (pandas.DataFrame): Data frame of feature values.
                Format: rows=records, col=features.
            ds_target (pandas.Series): Data series of target values.
                Format: rows=records, col=target.
            ds_weight (pandas.Series, optional, None): Data series of record
                weights. Format: rows=records, col=weight.
            cat_features (list, optional, None): List of column names that are
                categorical features. If default (`None`), uses
                `self.cat_features` from initialization. If defined, replaces
                `self.cat_features`. Example: ['ftr0', 'ftr1']
        
        Returns:
            None
        
        Attributes:
            feature_orig_med (collections.defaultdict): Nested dict as mapping
                of categorical feature to original values, then from original
                values to group's unweighted target median. Missing keys map to
                population's unweighted target median.
                Example: {'ftr0': {'a': 0, 'b': 1}, 'ftr1': {'A': 10, 'B': 20}}
        
        Raises:
            ValueError: Raised if some features in `cat_features` are not
                in `df_features.columns`.
        
        See Also:
            self.transform
        
        Notes:
            * For each categorical feature, group records by unique values and
                map the group values to the unweighted target median for that
                group.
            * For new data, values that have not been seen before are mapped
                to the target's unweighted median.

        """
        # Check arguments.
        if cat_features is not None:
            self.cat_features = cat_features
        else:
            cat_features = self.cat_features
        if not set(cat_features).issubset(df_features.columns):
            raise ValueError(
                "Some features in `cat_features` are not in `df_features.columns`\n" +
                "Required: set(cat_features).issubset(df_features.columns)")
        # Map categorical features to original values,
        # then from original values to group's unweighted target median.
        # Note: Compute the unweighted target median outside of the lambda func
        #     otherwise the median will be recomputed with every lambda call.
        if ds_weight is not None:
            target_vals = _unweight_target(
                ds_target=ds_target, ds_weight=ds_weight)
        else:
            target_vals = ds_target.values
        unweighted_target_median = np.nanmedian(target_vals)
        self.feature_orig_med = collections.defaultdict(
            lambda: unweighted_target_median)
        for feature in cat_features:
            self.feature_orig_med[feature] = collections.defaultdict(
                lambda: unweighted_target_median)
            for (feature_val, ds_target_group) in (
                ds_target.groupby(by=df_features[feature].values)):
                if ds_weight is not None:
                    ds_weight_group = ds_weight.loc[ds_target_group.index]
                    target_group_vals = _unweight_target(
                        ds_target=ds_target_group, ds_weight=ds_weight_group)
                else:
                    target_group_vals = ds_target_group.values
                unweighted_target_group_median = np.nanmedian(target_group_vals)
                self.feature_orig_med[feature][feature_val] = \
                    unweighted_target_group_median
        return None


    def transform(
        self, df_features:pd.DataFrame, ds_target:pd.Series,
        ds_weight:pd.Series=None, feature_orig_med:dict=None) -> tuple:
        r"""Transform the feature data frame by mapping categorical features
        to unweighted target medians by group. New columns with suffix '_med'
        are added to `df_features` from `feature_orig_med.keys()`.
        
        Args:
            self (implicit)
            df_features (pandas.DataFrame): Data frame of feature values.
                Format: rows=records, col=features.
            ds_target (pandas.Series): Data series of target values.
                Format: rows=records, col=target.
            ds_weight (pandas.Series, optional, None): Data series of record
                weights. Format: rows=records, col=weight.
            feature_orig_med (collections.defaultdict, optional, None): Nested
                dict as mapping of categorical feature to original values,
                then from original values to group's unweighted target median.
                Missing keys map to population's unweighted target median.
                Example: {'ftr0': {'a': 0, 'b': 1}, 'ftr1': {'A': 10, 'B': 20}}
                If default (`None`), must be previously set by `fit` method.
                See 'Raises'.
        
        Returns:
            df_features_tform (pandas.DataFrame): Data frame of features values
                with added features for categorical feature target medians,
                suffixed with '_med'. Format: rows=records, col=features.
            ds_target_tform (pandas.Series): Data series of target values,
                unchanged from input. Format: rows=records, col=target.
            ds_weight_tform (pandas.Series, optional, None): Data series of record
                weights, unchanged from input. Format: rows=records, col=weight.
        
        Raises:
            ValueError:
                * Raised if some features in `feature_orig_med` are not
                    in `df_features.columns`.
                * Raised if `feature_orig_med` is undefined,
                    i.e. if `feature_orig_med is None` and `fit` has not already
                    been called.
        
        See Also:
            self.fit, self.inverse_transform
        
        Notes:
            * `ds_target` and `ds_weight` are not used in `transform`.
                The arguments are passed to maintain a consistent API.

        """
        # Check arguments.
        if feature_orig_med is not None:
            if not set(feature_orig_med.keys()).issubset(df_features.columns):
                raise ValueError(
                    "Some features in `feature_orig_med` are not in `df_features.columns`\n" +
                    "Required: set(feature_orig_med.keys()).issubset(df_features.columns)")
            self.feature_orig_med = feature_orig_med
        if self.feature_orig_med is None:
            raise ValueError(
                "`feature_orig_med` is undefined. Assign `feature_orig_med`\n" +
                "or call `fit`.\n")
        # Copy data frames/series to avoid modifying input data.
        df_features_tform = df_features.copy()
        ds_target_tform = ds_target.copy()
        if ds_weight is not None:
            ds_weight_tform = ds_weight.copy()
        else:
            ds_weight_tform = ds_weight
        # Transform the feature values and add new features.
        for (feature, orig_med) in self.feature_orig_med.items():
            df_features_tform[feature+'_med'] =\
                df_features_tform[feature].replace(orig_med)
        return (df_features_tform, ds_target_tform, ds_weight_tform)
        
    
    def inverse_transform(
        self, df_features:pd.DataFrame, ds_target:pd.Series,
        ds_weight:pd.Series=None, feature_orig_med:dict=None) -> tuple:
        r"""Inverse transform the feature data frame by mapping grouped
        unweighted target medians to original categorical features. New columns
        with suffix '_med_orig' are added to `df_features` from
        `feature_orig_med.keys()`.
        
        Args:
            self (implicit)
            df_features (pandas.DataFrame): Data frame of feature values with
                categorical features mapped to target medians and suffixed
                with '_med'. Format: rows=records, col=features. See 'Raises'.
            ds_target (pandas.Series): Data series of target values.
                Format: rows=records, col=target.
            ds_weight (pandas.Series, optional, None): Data series of record
                weights. Format: rows=records, col=weight.
            feature_orig_med (collections.defaultdict, optional, None): Nested
                dict as mapping of categorical feature to original values,
                then from original values to group's unweighted target median.
                Missing keys map to population's unweighted target median.
                Example: {'ftr0': {'a': 0, 'b': 1}, 'ftr1': {'A': 10, 'B': 20}}
                If default (`None`), must be previously set by `fit` method.
                See 'Raises'.

        Returns:
            df_features_itform (pandas.DataFrame): Data frame of feature values
                with added features for original feature values,
                suffixed with '_med_orig'. Format: rows=records, col=features.
            ds_target_itform (pandas.Series): Data series of positive target
                values, unchanged from input. Format: rows=records, col=target.
            ds_weight_itform (pandas.Series, optional, None): Data series of
                record weights, unchanged from input.
                Format: rows=records, col=weight.
        
        Raises:
            ValueError:
                * Raised if some features in `feature_orig_med` with suffix
                    '_med' are not in `df_features.columns`.
                * Raised if `feature_orig_med` is undefined,
                    i.e. if `feature_orig_med is None` and `fit` has not already
                    been called.
        
        See Also:
            self.transform
        
        Notes:
            * `ds_target` and `ds_weight` are not used in `transform`.
                The arguments are passed to maintain a consistent API.
        
        """
        # Check arguments.
        if feature_orig_med is not None:
            features_med = [key+'_med' for key in feature_orig_med.keys()]
            if not set(features_med).issubset(df_features.columns):
                raise ValueError(
                    "Some features in `feature_orig_med` with suffix '_med'\n" +
                    "are not in `df_features.columns`\n" +
                    "Required: set(features_med).issubset(df_features.columns)\n" +
                    "where features_med = [key+'_med' for key in feature_orig_med.keys()]")
            self.feature_orig_med = feature_orig_med
        if self.feature_orig_med is None:
            raise ValueError(
                "`feature_orig_med` is undefined. Assign `feature_orig_med`\n" +
                "or call `fit`.\n")
        # Copy data frames/series to avoid modifying input data.
        df_features_itform = df_features.copy()
        ds_target_itform = ds_target.copy()
        if ds_weight is not None:
            ds_weight_itform = ds_weight.copy()
        else:
            ds_weight_itform = ds_weight
        # Transform the feature values and add new features.
        for (feature, orig_med) in self.feature_orig_med.items():
            med_orig = {val: key for (key, val) in orig_med.items()}
            df_features_itform[feature+'_med_orig'] =\
                df_features_itform[feature+'_med'].replace(med_orig)
        return (df_features_itform, ds_target_itform, ds_weight_itform)


def calc_silhouette_scores(
    df_features:pd.DataFrame, n_clusters_min:int=2, n_clusters_max:int=10,
    size_sub:int=None, n_scores:int=10,
    show_progress:bool=False, show_plot:bool=False) -> list:
    r"""Plot silhouette scores for determining number of clusters in k-means.
    
    Args:
        df_features (pandas.DataFrame): Data frame of feature values.
            Format: rows=records, cols=features.
            Note: `df_features` should aleady be scaled
            (e.g. sklearn.preprocessing.RobustScaler)
        n_clusters_min (int, optional, 2): Minimum number of clusters.
        n_clusters_max (int, optional, 10): Maximum number of clusters.
        size_sub (int, optional, None): Number of records in subset for
            calculating scores. See 'Notes', 'Raises'.
            Default: `None`, then min(1K, all) records are used.
        n_scores (int, optional, 10): Number of scores to calculate
            for each cluster. See 'Notes'.
        show_progress (bool, optional, False): Print status.
        show_plot (bool, optional, False): Show summary plot of scores.
        
    Returns:
        nclusters_scores (list): List of tuples (n_clusters, scores)
            where n_clusters is the number of clusters and
            scores are the calclated silhouette scores.
            Note: `sklearn.metrics.silhouette_score` may fail if cluster sizes
            are strongly imbalanced. In these cases,
            `len(scores) < n_scores` and the shape of `nclusters_scores` is
            irregular.

    Raises:
        ValueError:
            * Raised if not `2 <= n_clusters_min < n_clusters_max`.
        RuntimeWarning:
            * Raised if not `size_sub <= len(df_features)`.
    
    See Also:
        sklearn.cluster.MiniBatchKMeans,
        sklearn.metrics.silhouette_score
    
    Notes:
        * Silhouette scores are a measure comparing the relative size and
            proximity of clusters. Interpretation from [^sklearn]:
            "The best value is 1 and the worst value is -1. Values near 0
            indicate overlapping clusters. Negative values generally indicate
            that a sample has been assigned to the wrong cluster,
            as a different cluster is more similar."
        * For better score estimates, often it's more efficient to increase
            n_scores rather than size_sub since
            `sklearn.metrics.silhouette_score` creates a size_sub**2 matrix
            in RAM.
        * `sklearn.metrics.silhouette_score` may fail if cluster sizes
            are strongly imbalanced. In these cases,
            `len(scores) < n_scores` and the shape of `nclusters_scores` is
            irregular.
    
    References:
        [^sklearn] http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
    
    """
    # TODO: Replace show_progress and warnings.warn with logger.[debug,warn]
    #     https://github.com/stharrold/stharrold.github.io/issues/58
    # Check arguments.
    if not (2 <= n_clusters_min < n_clusters_max):
        raise ValueError(
            ("The number of clusters is not valid.\n" +
             "Required: 2 <= n_clusters_min < n_clusters_max\n" +
             "Given: 2 <= {nmin} < {nmax}").format(
                 nmin=n_clusters_min, nmax=n_clusters_max))
    if (size_sub is not None) and not (size_sub <= len(df_features)):
        warnings.warn(
            ("The number of records in the subset for calculating the\n" +
             "silhouette scores is larger than the number of records\n" +
             "in the data.\n" +
             "Suggested: size_sub <= len(df_features)\n" +
             "Given: {lhs} <= {rhs}").format(
                 lhs=size_sub, rhs=len(df_features)),
             RuntimeWarning)
    if size_sub is None:
        size_sub = min(int(1e3), len(df_features))
    # Estimate silhouette scores for each number of clusters.
    if show_progress:
        print("Progress:", end=' ')
    nclusters_scores = list()
    num_clusters = (n_clusters_max - n_clusters_min) + 1
    for n_clusters in range(n_clusters_min, n_clusters_max+1):
        transformer_kmeans = sk_cl.MiniBatchKMeans(n_clusters=n_clusters)
        labels_pred = transformer_kmeans.fit_predict(X=df_features)
        scores = list()
        n_fails = 0
        while len(scores) < n_scores:
            try:
                scores.append(
                    sk_met.silhouette_score(
                        X=df_features,
                        labels=labels_pred,
                        sample_size=size_sub))
            except ValueError:
                n_fails += 1
            if n_fails > 10*n_scores:
                warnings.warn(
                    ("`sklearn.silhouette_score` failed for given data with:\n" +
                     "n_clusters = {ncl}\n" +
                     "size_sub = {size}\n").format(
                         ncl=n_clusters, size=size_sub))
                break
        nclusters_scores.append((n_clusters, scores))
        if show_progress:
            print("{frac:.0%}".format(
                    frac=(n_clusters-n_clusters_min+1)/num_clusters),
                  end=' ')
    if show_progress:
        print('\n')
    # Plot silhouette scores vs number of clusters.
    if show_plot:
        nclusters_pctls = np.asarray(
            [np.append(tup[0], np.percentile(tup[1], q=[5,50,95]))
             for tup in nclusters_scores])
        plt.plot(
            nclusters_pctls[:, 0], nclusters_pctls[:, 2],
            marker='.', color=sns.color_palette()[0],
            label='50th pctl score')
        plt.fill_between(
            nclusters_pctls[:, 0],
            y1=nclusters_pctls[:, 1],
            y2=nclusters_pctls[:, 3],
            alpha=0.5, color=sns.color_palette()[0],
            label='5-95th pctls of scores')
        plt.title("Silhouette score vs number of clusters")
        plt.xlabel("Number of clusters")
        plt.ylabel("Silhouette score")
        plt.legend(loc='lower left')
        plt.show()
    return nclusters_scores


def calc_feature_importances(
    estimator,
    df_features:pd.DataFrame, ds_target:pd.Series, ds_weight:pd.Series=None, 
    size_sub:int=None, replace:bool=True, dists:list=None, 
    show_progress:bool=False, show_plot:bool=False) -> pd.DataFrame:
    r"""Calculate feature importances and compare to random added features
    for weighted data sets.
    
    Args:
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
            selecting features.
            Default: `None`, then min(10K, all) records are used.
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
        show_progress (bool, optional, False): Print status.
        show_plot (bool, optional, False): Show summary plot of max top 20 
            significant feature importances and the random importance.
        
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
        size_sub = min(int(1e4), size_data)
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
    # Return the feature importances and plot the 20 most important features.
    df_importances = pd.DataFrame.from_dict(ftrs_imps)
    if show_plot:
        ds_ftrs_imps_mean = df_importances.mean().sort_values(ascending=False)
        tfmask = (ds_ftrs_imps_mean > df_importances['random'].mean())
        ftrs_plot = list(ds_ftrs_imps_mean[tfmask].index[:20])+['random']
        sns.barplot(
            data=df_importances[ftrs_plot], order=ftrs_plot, ci=95,
            orient='h', color=sns.color_palette()[0])
        plt.title(
            ("Feature column name vs top 20 importance scores\n" +
             "with 95% confidence interval and benchmark randomized scores"))
        plt.xlabel(
            ("Importance score\n" +
             "(normalized reduction of loss function)"))
        plt.ylabel("Feature column name")
        plt.show()
    return df_importances


def calc_score_pvalue(
    estimator,
    df_features:pd.DataFrame, ds_target:pd.Series, ds_weight:pd.Series=None,
    n_iter:int=20, frac_true:float=0.2, size_sub:int=None, frac_test:float=0.2,
    replace:bool=True, show_progress:bool=False, show_plot:bool=False) -> float:
    r"""Calculate the p-value of the scored predictions for weighted data sets.
    
    Args:
        estimator: Estimator model from `sklearn` with attributes `fit`
            and, after running `fit`, also `score`. See 'Raises', 'Notes'.
        df_features (pandas.DataFrame): Data frame of feature values.
            Format: rows=records, cols=features.
        ds_target (pandas.Series): Data series of target values.
            Format: rows=records, col=target.
        ds_weight (pandas.Series, optional, None): Data series of record
            weights. Format: rows=records, col=weight.
            Default: `None`, then all set to 1.
        n_iter (int, optional, 20): Number of iterations for calculating scores.
        frac_true (float, optional, 0.2): Proportion of `n_iter` for which the
            target values are not shuffled. Must be between 0 and 1.
            See 'Raises'.
        size_sub (int, optional, None): Number of records in subset for
            cross-validating scores. Only enough records need to be used to
            approximate the variance in the data.
            Default: `None`, then min(10K, all) records are used. See 'Raises'.
        frac_test (float, optional, 0.2): Proportion of `size_sub` for which to
            test the predicted target values and calculate each score.
        replace (bool, optional, True): Sample records with replacement
            (bootstrap sampling). Default: `True`.
            Use `replace=True` to reduce overfitting the scores.
        show_progress (bool, optional, False): Print status.
        show_plot (bool, optional, False): Show summary plot
            of score significance.
    
    Returns:
        pvalue
    
    Raises:
        ValueError:
            * Raised if not `hasattr(estimator, 'fit')`
            * Raised if not `0.0 < frac_true < 1.0`.
        RuntimeWarning:
            * Raised if not `size_sub <= len(df_features)`.
    
    See Also:
        sklearn.cross_validation.train_test_split
    
    Notes:
        * The significance test is calculated by sampling the differences
            between the score means then shuffling the labels for whether or
            not the target values were themselves shuffled.
    
    References:
        [^sklearn]: http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.KFold.html

    """
    # TODO: Replace show_progress and warnings.warn with logger.[debug,warn]
    #     https://github.com/stharrold/stharrold.github.io/issues/58
    # Check arguments.
    if not hasattr(estimator, 'fit'):
        raise ValueError(
            ("`estimator` must have the attribute 'fit'.\n" +
             "Required: hasattr(estimator, 'fit')"))
    if not 0.0 < frac_true < 1.0:
        raise ValueError(
            ("`frac_true` must be between 0 and 1.\n" +
             "Required: 0.0 < frac_true < 1.0\n" +
             "Given: frac_true={frac_true}").format(frac_true=frac_true))
    if (size_sub is not None) and not (size_sub <= len(df_features)):
        warnings.warn(
            ("The number of records in the subset for calculating feature\n" +
             "importances is larger than the number of records in the data.\n" +
             "Suggested: size_sub <= len(df_features)\n" +
             "Given: {lhs} <= {rhs}").format(
                 lhs=size_sub, rhs=len(df_features)),
             RuntimeWarning)
    size_data = len(df_features)
    if size_sub is None:
        size_sub = min(int(1e4), size_data)
    # Score with/without shuffling the target values.
    inum_score_isshf = dict()
    imod = round(n_iter*frac_true)
    if show_progress:
        print("Progress:", end=' ')
    for inum in range(0, n_iter):
        inum_score_isshf[inum] = dict()
        idxs_sub = np.random.choice(a=size_data, size=size_sub, replace=replace)
        if inum % imod == 0:
            # Every 1 out of imod times, use true target values
            # and show progress.
            inum_score_isshf[inum]['is_shf'] = False
            trg_vals = ds_target.values[idxs_sub]
            if show_progress:
                print("{frac:.0%}".format(frac=(inum+1)/n_iter), end=' ')
        else:
            # Otherwise with randomly permuted target values.
            inum_score_isshf[inum]['is_shf'] = True
            trg_vals = np.random.permutation(ds_target.values[idxs_sub])
        if ds_weight is None:
            (ftrs_train, ftrs_test,
             trg_train, trg_test) = sk_cv.train_test_split(
                df_features.values[idxs_sub], trg_vals, test_size=frac_test)
            estimator.fit(X=ftrs_train, y=trg_train)
            inum_score_isshf[inum]['score'] = estimator.score(
                X=ftrs_test, y=trg_test)
        else:
            (ftrs_train, ftrs_test,
             trg_train, trg_test,
             pwt_train, pwt_test) = sk_cv.train_test_split(
                df_features.values[idxs_sub], trg_vals,
                ds_weight.values[idxs_sub], test_size=frac_test)
            estimator.fit(X=ftrs_train, y=trg_train, sample_weight=pwt_train)
            inum_score_isshf[inum]['score'] = estimator.score(
                X=ftrs_test, y=trg_test, sample_weight=pwt_test)
    if show_progress:
        print('\n')
    df_scores = pd.DataFrame.from_dict(data=inum_score_isshf, orient='index')
    # Plot the distributions of model scores with/without
    # shuffling the target values.
    if show_plot:
        sns.distplot(
            df_scores.loc[df_scores['is_shf'], 'score'],
            hist=True, kde=True, norm_hist=True, color=sns.color_palette()[0],
            label='scores with shuffled target values')
        sns.distplot(
            df_scores.loc[np.logical_not(df_scores['is_shf']), 'score'],
            hist=True, kde=True, norm_hist=True, color=sns.color_palette()[1],
            label='scores with actual target values')
        plt.title(
            "Probability density functions of model scores\n" +
            "by whether or not target values were permuted")
        plt.xlabel("Model score")
        plt.ylabel("Probability density")
        plt.legend(loc='upper left')
        plt.show()
        print("Average model score with shuffling: {score:.3f}".format(
                score=df_scores.loc[df_scores['is_shf'], 'score'].mean()))
        print("Average model score without shuffling: {score:.3f}".format(
                score=df_scores.loc[np.logical_not(df_scores['is_shf']), 'score'].mean()))
    # Calculate the distribution of differences in score means with/without
    # shuffling the target values.
    # Distribution of randomized differences in score means:
    rnd_mean_score_diffs = list()
    for _ in range(int(1e4)):
        tfmask_shf = np.random.permutation(df_scores['is_shf'].values)
        tfmask_notshf = np.logical_not(tfmask_shf)
        rnd_mean_score_diffs.append(
            df_scores.loc[tfmask_notshf, 'score'].mean()
            - df_scores.loc[tfmask_shf, 'score'].mean())
    # Distribution of actual differences in score means:
    tfmask_shf = df_scores['is_shf'].values
    tfmask_notshf = np.logical_not(tfmask_shf)
    mean_score_diff = (
        df_scores.loc[tfmask_notshf, 'score'].mean()
        - df_scores.loc[tfmask_shf, 'score'].mean())
    # Plot the distribution of differences in score means.
    if show_plot:
        sns.distplot(rnd_mean_score_diffs, hist=True, kde=True, norm_hist=True,
            color=sns.color_palette()[0],
            label=(
                'mean score differences assuming\n' +
                'no distinction between shuffled/unshuffled'))
        plt.axvline(mean_score_diff,
            color=sns.color_palette()[1], label='actual mean score difference')
        plt.title(
            "Differences between mean model scores\n" +
            "by whether or not target values were actually shuffled")
        plt.xlabel("Model score difference")
        plt.ylabel("Probability density")
        plt.legend(loc='upper left')
        plt.show()
    # Return the p-value and describe the statistical significance.
    pvalue = 100.0 - scipy.stats.percentileofscore(
        a=rnd_mean_score_diffs, score=mean_score_diff, kind='mean')
    if show_plot:
        print(
            ("Null hypothesis: There is no distinction in the differences\n" +
             "between the mean model scores whether or not the target\n" +
             "values have been shuffled.\n" +
             "Outcome: Assuming the null hypothesis, the probability of\n" +
             "obtaining a difference between the mean model scores at least\n" +
             "as great as {diff:.2f} is {pvalue:.1f}%.").format(
                 diff=mean_score_diff, pvalue=pvalue))
    return pvalue


def plot_actual_vs_predicted(
    y_true:np.ndarray, y_pred:np.ndarray, loglog:bool=False, xylims:tuple=None,
    path:str=None) -> None:
    r"""Plot actual vs predicted values.
    
    Args:
        y_true (numpy.ndarray): The true target values.
        y_pred (numpy.ndarray): The predicted target values.
        loglog (bool, optional, False): Log scale for both x and y axes.
        xylims (tuple, optional, None): Limits for x and y axes.
            Default: None, then set to (min, max) of `y_pred`.
        path (str, optional, None): Path to save figure.
    
    Returns:
        None
    
    """
    # TODO: Plot binned percentiles; Q-Q plot
    # TODO: Z1,Z2 gaussianity measures
    # Check input.
    # TODO: limit number of points to plot
    # TODO: Use hexbins for density.
    #   sns.jointplot(
    #       x=y_pred, y=y_true, kind='hex', stat_func=None,
    #       label='(predicted, actual)')
    plt.title("Actual vs predicted values")
    if loglog:
        plot_func = plt.loglog
        y_pred_extrema = (min(y_pred[y_pred > 0]), max(y_pred))
    else:
        plot_func = plt.plot
        y_pred_extrema = (min(y_pred), max(y_pred))
    if xylims is not None:
        y_pred_extrema = xylims
    plot_func(
        y_pred, y_true, color=sns.color_palette()[0],
        marker='.', linestyle='', alpha=0.1, label='(predicted, actual)')
    plot_func(
        y_pred_extrema, y_pred_extrema, color=sns.color_palette()[1],
        marker='', linestyle='-', linewidth=1, label='(predicted, predicted)')
    plt.xlabel("Predicted values")
    plt.xlim(y_pred_extrema)
    plt.ylabel("Actual values")
    plt.ylim(y_pred_extrema)
    plt.legend(loc='upper left', title='values')
    if path is not None:
        plt.savefig(path)
    plt.show()
    return None
