#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Utilities for machine learning.

"""


# Import standard packages.
import collections
import itertools
import pdb
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
    

class StepReplaceTargetWithNormal:
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
            >>> cls = StepReplaceTargetWithNormal()
            >>> cls.fit(ftrs, trg, wt)
            >>> (ftrs_tform, trg_tform, wt_tform) =\
            ... cls.transform(ftrs, trg, wt)
            >>> (ftrs_itform, trg_itform, wt_itform) =\
            ... cls.inverse_transform(ftrs_tform, trg_tform, wt_tform)
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
        r"""Pseudo-private method to initialize class.
        
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
            self.lmbda (float): Calculated by `fit`. Optimal transformation
                parameter found by `scipy.stats.boxcox_normmax`. See 'Notes'.
            
        See Also:
            self.__init__, self.transform, scipy.stats.boxcox_normmax
        
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
        ds_weight:pd.Series=None) -> tuple:
        r"""Transform data frames/series by keeping only records with positive
        target values and transforming unweighted target values to be normally
        distributed using a Box-Cox power transformation.
        
        Args:
            self (implicit)
            df_features (pandas.DataFrame): Data frame of feature values.
                Format: rows=records, col=features.
            ds_target (pandas.Series): Data series of target values.
                Format: rows=records, col=target.
            ds_weight (pandas.Series, optional, None): Data series of record
                weights. Format: rows=records, col=weight.

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
            ValueError: Raised if `self.lmbda` is undefined,
                i.e. if `lmbda` was not assigned with the class initialization
                or `fit` has not already been called.
        
        See Also:
            self.fit, self.inverse_transform, scipy.stats.boxcox
        
        Notes:
            * Box-Cox transformation:[^wiki]
                y = (x^lam - 1)/lam    , if lam != 0
                y = ln(x)              , if lam = 0
        
        """
        # Check arguments.
        if self.lmbda is None:
            raise ValueError(
                "`self.lmbda` is undefined. Call `fit` to calculate.")
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
        return (df_features_tform, ds_target_tform, ds_weight_tform)


    @staticmethod
    def _inverse_boxcox(y_vals, lmbda:float):
        r"""Pseudo-private method to invert `scipy.stats.boxcox`.
        
        Args:
            y_vals (float or numpy.ndarray): Box-Cox transformed values to
                inverse transform.
            lmbda (float): Box-Cox transformation parameter.
        
        Returns:
            x_vals (float or numpy.ndarray): Inverse transformed values with
                same shape as `y_vals`.
        
        See Also:
            self.inverse_transform, scipy.stats.boxcox
        
        Notes:
            * Inverse Box-Cox transformation:
                x = (y*lam + 1)^(1/lam), if lam != 0
                x = exp(y)             , if lam = 0
            * As a static method, the call signature forces arguments to be
                passed explicitly.
        
        """
        y_vals = np.asarray(y_vals)
        if lmbda == 0:
            x_vals = np.exp(y_vals)
        else:
            x_vals = (y_vals*lmbda + 1)**(1/lmbda)
        return x_vals


    def inverse_transform(
        self, df_features:pd.DataFrame, ds_target:pd.Series,
        ds_weight:pd.Series=None) -> tuple:
        r"""Inverse transform target values from normal distribution to original
        distribution by inverted Box-Cox power transformation.
        
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
                * Raised if `self.lmbda` is undefined, i.e. if `lmbda` was not
                    initialized with the class or `fit` has not already
                    been called.
        
        See Also:
            self.fit, self.transform, scipy.stats.boxcox
        
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
        if self.lmbda is None:
            raise ValueError(
                "`self.lmbda` is undefined. Call `fit` to calculate.")
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
        return (df_features_itform, ds_target_itform, ds_weight_itform)


class StepDropConstantFeatures:
    r"""Pipeline step: Remove features that are constants.
    
    Notes:
        * Motivation: This transformation removes features that have only a
            single value for all records, and are thus uninformative.
        * Transformed/inverse transformed feature data frames have
            dropped/added columns.
        
    Examples:
        * Example fit, transform, and inverse transform:
            Constant features, ['ftr0', 'ftr1'], are transformed then inverse
            transformed. The transformed/inverse transformed features are
            dropped/added. The features added from the inverse transformation
            may not be in the original order.
            ```
            >>> const_features = ['ftr0', 'ftr1']
            >>> (ftrs, trg, wt) = (df_features, ds_target, ds_weight)
            >>> cls = StepDropConstantFeatures()
            >>> cls.fit(ftrs, trg, wt)
            >>> (ftrs_tform, trg_tform, wt_form) =\
            ... cls.transform(ftrs, trg, wt)
            >>> len(ftrs.columns) < len(ftrs_tform.columns)
            True
            >>> (ftrs_itform, trg_itform, wt_itform) =\
            ... cls.inverse_transform(ftrs_tform, trg_tform, wt_tform)
            >>> np.all(ftrs == ftrs_itform[ftrs.columns])
            True
            ```
    
    See Also:
        sklearn.pipeline.Pipeline
    
    """
    
    
    def __init__(self) -> None:
        r"""Pseudo-private method to initialize class.
        
        Args:
            self (implicit)
            
        Returns:
            None

        Attributes:
            features_const (dict): Dict mapping column names to that column's
                constant value. Initialized to `None`. Calcuate using `fit`
                method. Example: {'ftr0': 0, 'ftr1': 1}

        See Also:
            self.fit

        """
        self.features_const = None
        return None
        
    
    def fit(
        self, df_features:pd.DataFrame, ds_target:pd.Series,
        ds_weight:pd.Series=None) -> None:
        r"""Fit to determine constant features.

        Args:
            self (implicit)
            df_features (pandas.DataFrame): Data frame of feature values.
                Format: rows=records, col=features.
            ds_target (pandas.Series): Data series of target values.
                Format: rows=records, col=target.
            ds_weight (pandas.Series, optional, None): Data series of record
                weights. Format: rows=records, col=weight.
        
        Returns:
            None
            
        Attributes:
            self.features_const (dict): Calculated by `fit`. Dict mapping
                column names to that column's constant value.
                Example: {'ftr0': 0, 'ftr1': 1}
        
        Raises:
            ValueError: Raised if some features in `self.const_features` defined
                during class initialization are not in `df_features.columns`.
        
        See Also:
            self.__init__, self.transform
        
        Notes:
            * Null values are treated as distinct values. A feature with
                null and non-null values is not considered a constant.
            * `ds_target` and `ds_weight` are not used in `fit`.
                The arguments are passed to maintain a consistent API.
            
        """
        # Determine which features are constant.
        self.features_const = dict()
        for feature in df_features.columns:
            first_value = df_features[feature].iloc[0]
            if np.all(df_features[feature] == first_value):
                self.features_const[feature] = first_value
        return None


    def transform(
        self, df_features:pd.DataFrame, ds_target:pd.Series,
        ds_weight:pd.Series=None) -> tuple:
        r"""Transform the feature data frame by dropping the constant features.

        Args:
            self (implicit)
            df_features (pandas.DataFrame): Data frame of feature values.
                Format: rows=records, col=features.
            ds_target (pandas.Series): Data series of target values.
                Format: rows=records, col=target.
            ds_weight (pandas.Series, optional, None): Data series of record
                weights. Format: rows=records, col=weight.

        Returns:
            df_features_tform (pandas.DataFrame): Data frame of features values
                with constant features dropped.
                Format: rows=records, col=features.
            ds_target_tform (pandas.Series): Data series of target values,
                unchanged from input. Format: rows=records, col=target.
            ds_weight_tform (pandas.Series, optional, None): Data series of
                record weights, unchanged from input.
                Format: rows=records, col=weight.
        
        Raises:
            ValueError:
                * Raised if `self.features_const` is undefined,
                    i.e. if `fit` has not already been called.

        See Also:
            self.fit, self.inverse_transform
        
        Notes:
            * `ds_target` and `ds_weight` are not used in `transform`.
                The arguments are passed to maintain a consistent API.

        """
        # Check arguments.
        if self.features_const is None:
            raise ValueError(
                "`self.features_const` is undefined. Call `fit` to calculate.")
        # Copy data frames/series to avoid modifying input data.
        df_features_tform = df_features.copy()
        ds_target_tform = ds_target.copy()
        if ds_weight is not None:
            ds_weight_tform = ds_weight.copy()
        else:
            ds_weight_tform = ds_weight
        # Drop constant features.
        df_features_tform.drop(
            labels=self.features_const.keys(), axis=1, inplace=True)
        return (df_features_tform, ds_target_tform, ds_weight_tform)


    def inverse_transform(
        self, df_features:pd.DataFrame, ds_target:pd.Series,
        ds_weight:pd.Series=None) -> tuple:
        r"""Inverse transform the feature data frame by adding the dropped
        constant features.

        Args:
            self (implicit)
            df_features (pandas.DataFrame): Data frame of features values
                with constant features dropped.
                Format: rows=records, col=features.
            ds_target (pandas.Series): Data series of target values.
                Format: rows=records, col=target.
            ds_weight (pandas.Series, optional, None): Data series of record
                weights. Format: rows=records, col=weight.

        Returns:
            df_features_itform (pandas.DataFrame): Data frame of features values
                with constant features added.
                Format: rows=records, col=features.
            ds_target_itform (pandas.Series): Data series of target values,
                unchanged from input. Format: rows=records, col=target.
            ds_weight_itform (pandas.Series, optional, None): Data series of
                record weights, unchanged from input.
                Format: rows=records, col=weight.
        
        Raises:
            ValueError:
                * Raised if `self.features_const` is undefined,
                    i.e. if `fit` has not already been called.

        See Also:
            self.fit, self.transform
        
        Notes:
            * `ds_target` and `ds_weight` are not used in `transform`.
                The arguments are passed to maintain a consistent API.

        """
        # Check arguments.
        if self.features_const is None:
            raise ValueError(
                "`self.features_const` is undefined. Call `fit` to calculate.")
        # Copy data frames/series to avoid modifying input data.
        df_features_itform = df_features.copy()
        ds_target_itform = ds_target.copy()
        if ds_weight is not None:
            ds_weight_itform = ds_weight.copy()
        else:
            ds_weight_itform = ds_weight
        # Add constant features.
        for (feature, const) in self.features_const.items():
            df_features_itform[feature] = const
        return (df_features_itform, ds_target_itform, ds_weight_itform)
    

class StepReplaceCategoricalWithTarget:
    r"""Pipeline step: Replace categorical features with group-wise target
    means.
    
    Notes:
        * Motivation: This transformation includes an informative prior into
            the feature, i.e. the feature's group-wise relationship to the
            target. Splits in the transformed feature now correspond to splits
            in the target value. The mean is used instead of the median to
            ensure that most features are mapped to unique target values,
            i.e. a 1-1 mapping with a well-defined inverse.
        * For the inverse transformation, if the mapping between categorical
            features and target means is not a 1-1 mapping, then each unique
            target mean maps to only one value within each categorical feature.
        * Transformed/inverse transformed feature columns are overwritten.

    Examples:
        * Example fit, transform, and inverse transform:
            Categorical features, ['ftr0', 'ftr1'], are transformed then inverse
            transformed. The transformed/inverse transformed features
            are overwritten.
            ```
            >>> cat_features = ['ftr0', 'ftr1']
            >>> (ftrs, trg, wt) = (df_features, ds_target, ds_weight)
            >>> cls = StepReplaceCategoricalWithTarget(cat_features)
            >>> cls.fit(ftrs, trg, wt)
            >>> (ftrs_tform, trg_tform, wt_tform) =\
            ... cls.transform(ftrs, trg, wt)
            >>> ftrs.shape == ftrs_tform.shape
            True
            >>> (ftrs_itform, trg_itform, wt_itform) =\
            ... cls.inverse_transform(ftrs_tform, trg_tform, wt_tform)
            >>> # transformation is invertible only if mappings are 1-1:
            >>> np.all(ftrs == ftrs_itform)
            True
            ```

    See Also:
        sklearn.pipeline.Pipeline
    
    """
    

    def _calc_features_mean_orig(
        self, default_factory=None,
        warn:bool=False) -> collections.defaultdict:
        r"""Pseudo-private method to calculate `self.features_mean_orig` by
        inverting `self.features_orig_mean`.
        
        Args:
            self (implicit)
            default_factory (function, optional, None): Factory function to
                set default values of `self.features_mean_orig`.
                If default (`None`),
                `default_factory = self.features_orig_mean.default_factory`
                for top-level and
                `default_factory = self.features_orig_mean[feature].default_factory`
                for bottom-level.
            warn (bool, optional, False): Warn if `self.features_orig_mean`
                is not a 1-1 mapping, i.e. two unique values within a
                categorical feature map to the same target mean. See 'Raises'.
        
        Returns:
            None
        
        Attributes:
            self.features_mean_orig (collections.defaultdict): Calculated by
                `_calc_features_mean_orig`. Nested default dict as mapping
                of categorical feature to group's unweighted target mean then
                to original values. Missing keys map to population's unweighted
                target mean. Duplicate 
                Example: {'ftr0': {0: 'a', 1: 'b'}, 'ftr1': {10: 'A', 20: 'B'}}

        Raises:
            ValueError: Raised if `self.features_orig_mean` is undefined,
                i.e. if `features_orig_mean` was not assigned with the class
                initialization or if `fit` has not already been called.
            RuntimeWarning: Raised if `warn=True` and `self.features_orig_mean`
                is not a 1-1 mapping, i.e. two unique values within a
                categorical feature map to the same target mean. Only the first
                value seen is defined. All other values are ignored.
                Example: {'ftr0': {'a': 1, 'b': 1}} would raise RuntimeWaring.

        See Also:
            self.__init__, self.fit
        
        """
        # Check arguments.
        if self.features_orig_mean is None:
            raise ValueError(
                "`self.features_orig_mean` is undefined. Call `fit` to calculate.")
        # Calculate self.features_mean_orig.
        if default_factory is None:
            default_factory0 = self.features_orig_mean.default_factory
        else:
            default_factory0 = default_factory
        self.features_mean_orig = collections.defaultdict(default_factory0)
        for (feature, orig_mean) in self.features_orig_mean.items():
            if default_factory is None:
                default_factory1 = orig_mean.default_factory
            else:
                default_factory1 = default_factory
            self.features_mean_orig[feature] = collections.defaultdict(default_factory1)
            for (orig, mean) in orig_mean.items():
                if mean in self.features_mean_orig[feature].keys():
                    if warn:
                        warnings.warn(
                            ("`self.features_orig_mean` mapping is not unique. Inverse is ill-defined.\n" +
                             "Defined: self.features_mean_orig[{ftr}][{mean}] = {orig_def}\n" +
                             "Ignored: {{{ftr}: {{{mean}: {orig_ig}}}}}").format(
                                 ftr=feature, mean=mean,
                                 orig_def=self.features_mean_orig[feature][mean],
                                 orig_ig=orig),
                            RuntimeWarning)
                else:
                    self.features_mean_orig[feature][mean] = orig
        return None

    
    def __init__(
        self, cat_features:list,
        features_orig_mean:collections.defaultdict=None,
        warn:bool=False) -> None:
        r"""Pseudo-private method to initialize class.
        
        Args:
            self (implicit)
            cat_features (list): List of column names that are categorical
                features. Example: ['ftr0', 'ftr1']
            features_orig_mean (collections.defaultdict, optional, None): Nested
                dict as mapping of categorical feature to original values,
                then from original values to group's unweighted target mean.
                Missing keys map to population's unweighted target mean.
                If default (`None`), compute using `fit` method.
                Example: {'ftr0': {'a': 0, 'b': 1}, 'ftr1': {'A': 10, 'B': 20}}
            warn (bool, optional, False): Warn if `features_orig_mean` is not a
                1-1 mapping, i.e. two unique values within a categorical feature
                map to the same target mean. See 'Raises'.

        Returns:
            None

        Attributes:
            self.features_mean_orig (collections.defaultdict): Calculated if
                `features_orig_mean` assigned, otherwise initialized to `None`.
                Nested dict as mapping of categorical feature to group's
                unweighted target mean then to original values. Missing keys 
                map to `None`.
                Example: {'ftr0': {0: 'a', 1: 'b'}, 'ftr1': {10: 'A', 20: 'B'}}

        Raises:
            ValueError: Raised if some features in `cat_features` are
                not in `features_orig_mean`.
            RuntimeWarning: Raised if `warn=True` and `features_orig_mean` is
                not a 1-1 mapping, i.e. two unique values within a categorical
                feature map to the same target mean. Only the first value seen
                is defined. All other values are ignored.
                Example: {'ftr0': {'a': 1, 'b': 1}} would raise RuntimeWaring.
            
        See Also:
            self.fit
        
        """
        # Check arguments.
        if features_orig_mean is not None:
            if not set(cat_features).issubset(features_orig_mean.keys()):
                raise ValueError(
                    "Some features in `cat_features` are not in `features_orig_mean`\n" +
                    "Required: set(cat_features).issubset(features_orig_mean.keys())")
        # Assign arguments as instance attributes.
        self.cat_features = cat_features
        self.features_orig_mean = features_orig_mean
        self.features_mean_orig = None
        if self.features_orig_mean is not None:
            self._calc_features_mean(default_factory=lambda: None, warn=True)
        return None
    
    
    def fit(
        self, df_features:pd.DataFrame, ds_target:pd.Series,
        ds_weight:pd.Series=None, warn:bool=False) -> None:
        r"""Fit categorical features to unweighted target means by group.

        Args:
            self (implicit)
            df_features (pandas.DataFrame): Data frame of feature values.
                Format: rows=records, col=features.
            ds_target (pandas.Series): Data series of target values.
                Format: rows=records, col=target.
            ds_weight (pandas.Series, optional, None): Data series of record
                weights. Format: rows=records, col=weight.
            warn (bool, optional, False): Warn if `self.features_orig_mean`
                is not a 1-1 mapping, i.e. two unique values within a
                categorical feature map to the same target mean. See 'Raises'.

        Returns:
            None
        
        Attributes:
            self.features_orig_mean (collections.defaultdict): Calculated by
                `fit`. Nested dict as mapping of categorical feature to original
                values, then from original values to group's unweighted target
                mean. Missing keys map to population's unweighted target mean.
                Example: {'ftr0': {'a': 0, 'b': 1}, 'ftr1': {'A': 10, 'B': 20}}
            self.features_mean_orig (collections.defaultdict): Calculated by
                `fit`. Nested dict as mapping of categorical feature to group's
                unweighted target mean then to original values. Missing keys
                map to `None`.
                Example: {'ftr0': {0: 'a', 1: 'b'}, 'ftr1': {10: 'A', 20: 'B'}}
        
        Raises:
            ValueError: Raised if some features in `self.cat_features` are not
                in `df_features.columns`.
            RuntimeWarning: Raised if `warn=True` and `self.features_orig_mean`
                is not a 1-1 mapping, i.e. two unique values within a
                categorical feature map to the same target mean. Only the first
                value seen is defined. All other values are ignored.
                Example: {'ftr0': {'a': 1, 'b': 1}} would raise RuntimeWaring.
        
        See Also:
            self.__init__, self.transform
        
        """
        # Check arguments.
        if not set(self.cat_features).issubset(df_features.columns):
            raise ValueError(
                "Some features in `self.cat_features` are not in `df_features.columns`\n" +
                "Required: set(self.cat_features).issubset(df_features.columns)")
        # Map categorical features to original values,
        # then from original values to group's unweighted target mean.
        # Note: Compute the unweighted target mean outside of the lambda
        #     function otherwise the mean will be recomputed with every
        #     call of the lambda function.
        if ds_weight is not None:
            target_vals = _unweight_target(
                ds_target=ds_target, ds_weight=ds_weight)
        else:
            target_vals = ds_target.values
        unweighted_target_mean = np.nanmean(target_vals)
        self.features_orig_mean = collections.defaultdict(
            lambda: unweighted_target_mean)
        for feature in self.cat_features:
            self.features_orig_mean[feature] = collections.defaultdict(
                lambda: unweighted_target_mean)
            for (feature_val, ds_target_group) in (
                ds_target.groupby(by=df_features[feature].values)):
                if ds_weight is not None:
                    ds_weight_group = ds_weight.loc[ds_target_group.index]
                    target_group_vals = _unweight_target(
                        ds_target=ds_target_group, ds_weight=ds_weight_group)
                else:
                    target_group_vals = ds_target_group.values
                unweighted_target_group_mean = np.nanmean(target_group_vals)
                self.features_orig_mean[feature][feature_val] =\
                    unweighted_target_group_mean
        # Make the inverse mapping:
        #     Map categorical features to group's unweighted target mean,
        #     then from the target mean to the group's original value.
        self._calc_features_mean_orig(default_factory=lambda: None, warn=warn)
        return None


    def transform(
        self, df_features:pd.DataFrame, ds_target:pd.Series,
        ds_weight:pd.Series=None) -> tuple:
        r"""Transform the feature data frame by mapping categorical features
        to unweighted target means by group. Transformed features are
        replaced.
        
        Args:
            self (implicit)
            df_features (pandas.DataFrame): Data frame of feature values.
                Format: rows=records, col=features.
            ds_target (pandas.Series): Data series of target values.
                Format: rows=records, col=target.
            ds_weight (pandas.Series, optional, None): Data series of record
                weights. Format: rows=records, col=weight.

        Returns:
            df_features_tform (pandas.DataFrame): Data frame of features values
                with categorical features mapped to target means.
                Format: rows=records, col=features.
            ds_target_tform (pandas.Series): Data series of target values,
                unchanged from input. Format: rows=records, col=target.
            ds_weight_tform (pandas.Series, optional, None): Data series of
                record weights, unchanged from input.
                Format: rows=records, col=weight.
        
        Raises:
            ValueError:
                * Raised if `self.features_orig_mean` is undefined,
                    i.e. if `features_orig_mean` was not assigned with the class
                    initialization or if `fit` has not already been called.
                * Raised if some features in `self.features_orig_mean` are not
                    in `df_features.columns`.
        
        See Also:
            self.fit, self.inverse_transform
        
        Notes:
            * `ds_target` and `ds_weight` are not used in `transform`.
                The arguments are passed to maintain a consistent API.

        """
        # Check arguments.
        if self.features_orig_mean is None:
            raise ValueError(
                "`self.features_orig_mean` is undefined. Call `fit` to calculate.")
        if not set(self.features_orig_mean.keys()).issubset(df_features.columns):
            raise ValueError(
                "Some features in `self.features_orig_mean` are not in `df_features.columns`\n" +
                "Re-run `self.fit` to rebuild `self.features_orig_mean`.\n" +
                "Required: set(self.features_orig_mean.keys()).issubset(df_features.columns)")
        # Copy data frames/series to avoid modifying input data.
        df_features_tform = df_features.copy()
        ds_target_tform = ds_target.copy()
        if ds_weight is not None:
            ds_weight_tform = ds_weight.copy()
        else:
            ds_weight_tform = ds_weight
        # Transform the feature values
        # from their original values to grouped target means.
        df_features_tform.replace(self.features_orig_mean, inplace=True)
        return (df_features_tform, ds_target_tform, ds_weight_tform)
        
    
    def inverse_transform(
        self, df_features:pd.DataFrame, ds_target:pd.Series,
        ds_weight:pd.Series=None) -> tuple:
        r"""Inverse transform the feature data frame by mapping grouped
        unweighted target means to original categorical features. Inverse
        transformed features are replaced.
        
        Args:
            self (implicit)
            df_features (pandas.DataFrame): Data frame of feature values with
                categorical features mapped to target means.
                Format: rows=records, col=features. See 'Raises'.
            ds_target (pandas.Series): Data series of target values.
                Format: rows=records, col=target.
            ds_weight (pandas.Series, optional, None): Data series of record
                weights. Format: rows=records, col=weight.

        Returns:
            df_features_itform (pandas.DataFrame): Data frame of feature values
                with categorical features mapped to original feature values.
                Format: rows=records, col=features.
            ds_target_itform (pandas.Series): Data series of target values,
                unchanged from input. Format: rows=records, col=target.
            ds_weight_itform (pandas.Series, optional, None): Data series of
                record weights, unchanged from input.
                Format: rows=records, col=weight.
        
        Raises:
            ValueError:
                * Raised if `self.features_mean_orig` is undefined,
                    i.e. if `features_orig_mean` was not assigned with the class
                    initialization or if `fit` has not already been called.
                * Raised if some features in `self.features_orig_mean` are not
                    in `df_features.columns`.
        
        See Also:
            self.fit, self.transform
        
        Notes:
            * `ds_target` and `ds_weight` are not used in `transform`.
                The arguments are passed to maintain a consistent API.
        
        """
        # Check arguments.
        if self.features_orig_mean is None:
            raise ValueError(
                "`self.features_orig_mean` is undefined. Call `fit` to calculate.")
        if not set(self.features_orig_mean.keys()).issubset(df_features.columns):
            raise ValueError(
                "Some features in `self.features_orig_mean` are not in `df_features.columns`\n" +
                "Required: set(self.features_orig_mean.keys()).issubset(df_features.columns)")
        if self.features_mean_orig is None:
            raise AssertionError(
                "Program error. `self.features_mean_orig` is not defined.\n" +
                "Required: if `self.features_orig_mean is not None`,\n" +
                "then `self.features_mean_orig` should already be calculated.")
        # Copy data frames/series to avoid modifying input data.
        df_features_itform = df_features.copy()
        ds_target_itform = ds_target.copy()
        if ds_weight is not None:
            ds_weight_itform = ds_weight.copy()
        else:
            ds_weight_itform = ds_weight
        # Inverse transform the feature values
        # from grouped target means to their original values.
        df_features_itform.replace(self.features_mean_orig, inplace=True)
        return (df_features_itform, ds_target_itform, ds_weight_itform)


class StepRobustScaleFeatures:
    r"""Pipeline step: Scale feature values using statistics that are robust
    to outliers. Record weights are not used.
    
    Notes:
        * Motivation: PCA and k-means clustering perform better when the
            feature values have a standard scale.
        * Because `sklearn.preprocessing.RobustScaler` uses the interquartile
            range (25th-75th percentiles) as the measure of the shape of
            the feature distibution, the interquartile range of the transformed
            feature distribution will be either 1 (if the original 25th and 75th
            percentiles were not equal) or 0 (if they were equal).
        * Transformed/inverse transformed features replace original features.
    
    Examples:
        * Example fit, transform, and inverse transform:
            Features are replaced with scaled features.
            ```
            >>> (ftrs, trg, wt) = (df_features, ds_target, ds_weight)
            >>> cls = StepRobustScaleFeatures()
            >>> cls.fit(ftrs, trg, wt)
            >>> (ftrs_tform, trg_tform, wt_tform) =\
            ... cls.transform(ftrs, trg, wt)
            >>> ftrs_tform.shape == ftrs.shape
            True
            >>> np.all(np.isclose(ftrs_tform.median(axis=0), 0))
            True
            >>> (ftrs_itform, trg_itform, wt_itform) =\
            ... cls.inverse_transform(ftrs_tform, trg_tform, wt_tform)
            >>> np.all(np.isclose(ftrs, ftrs_itform))
            True
            ```
    
    See Also:
        sklearn.pipeline.Pipeline, sklearn.preprocessing.RobustScaler
    
    """
    
    
    def __init__(self, subset_features:list=None) -> None:
        r"""Pseudo-private method to initialize class.
        
        Args:
            self (implicit)
            subset_features (list, optional, None): List of feature column names
                to robustly scale. If default (`None`), all feature columns
                are scaled. Example: ['ftr0', 'ftr1']
        
        Returns:
            None
        
        See Also:
            self.fit
        
        """
        self.subset_features = subset_features
        return None
        
        
        def fit(
            self, df_features:pd.DataFrame, ds_target:pd.Series,
            ds_weight:pd.Series=None) -> None:
            r"""Fit a robust scaler to features. Record weights are not used.
            
            Args:
                self (implicit)
                df_features (pandas.DataFrame): Data frame of feature values.
                    Format: rows=records, col=features.
                ds_target (pandas.Series): Data series of target values.
                    Format: rows=records, col=target.
                ds_weight (pandas.Series, optional, None): Data series of record
                    weights. Format: rows=records, col=weight.

            Returns:
                None
            
            Raises:
                ValueError: Raised if some features in `self.subset_features`
                    defined during class initialization are not in
                    `df_features.columns`.
            
            See Also:
                self.__init__, self.transform
                
            Notes:
                * `ds_target` and `ds_weight` are not used in `fit`.
                    The arguments are passed to maintain a consistent API.
            
            """
            # Check arguments.
            if subset_features is not None:
                self.subset_features = subset_features
            else:
                #subset_features = 
                pass
            return None


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
