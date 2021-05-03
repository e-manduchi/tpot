# -*- coding: utf-8 -*-

"""This file is part of the TPOT library.

TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - and many more generous open source contributors

TPOT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

TPOT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with TPOT. If not, see <http://www.gnu.org/licenses/>.

Scope
-----
Transforms specified features by residual covariate adjustment for 
specified covariates. Removes covariate columns, but retains indicator
and adjY columns (when present).

"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression, LogisticRegression
import re

def X_adj_fit(X, C, col_type):
    """Fits a col_type regression estimator to target X on features C corresponding to covariates.

    Parameters
    ----------
    X : pd.Series
    	feature to use as target for the estimator.
    C: pd.DataFrame of covariates
    	covariats to use as features for the estimator.
    col_type: 'logistic' or 'linear'
    	depending on X type.

    Returns
    -------
    est: estimator
    	fitted col_type regression estimator.

    """

    X_col = X.values # 1D np.ndarray
    if col_type == 'linear':
        est = LinearRegression()
        est.fit(C, X_col)
       # To do: handle no convergence case

    elif col_type == 'logistic':
        est = LogisticRegression(penalty='none',
                                 solver='lbfgs',
                                 multi_class='auto',
                                 max_iter=500)
        est.fit(C, X_col.astype(np.int32))
       # To do: handle no convergence case

    else:
        raise ValueError("Wrong column type! It should be 'logistic' or 'linear'!")

    return est

def X_adj_predict(X, C, col_type, est):
    """Transform feature X using residual adjustments from estimator est on covariates C.

    Parameters
    ----------
    X : pd.Series
    	feature to adjust.
    C: pd.DataFrame
    	covariates to adjust by.
    col_type: 'logistic' or 'linear'
    	depending on X.
    est: estimator
    	fitted col_type regression estimator for X on C.

    Returns
    -------
    X_adj: transformed X
    	by residual adjustment.

    """

    X_col = X.values # 1D np.ndarray
    if col_type == 'linear':
        est_pred = est.predict(C)
        X_adj = X_col - est_pred

    elif col_type == 'logistic':
        clf_pred_proba = est.predict_proba(C)
        X_adj = X_col
        for gt_idx, gt in enumerate(est.classes_):
            gt = int(gt)
            X_adj = X_adj - gt*clf_pred_proba[:, gt_idx]
    else:
        raise ValueError("Wrong column type! It should be 'logistic' or 'linear'!")

    return X_adj.reshape(-1, 1)

class resAdjTransformer(BaseEstimator, TransformerMixin):
    """Transformer by residual adjustments."""

    def __init__(self, C=None, adj_list=None):
        """Iniialization.

        Parameters
        ----------
        C: a list of columns for C, e.g ['cov4', 'cov7', 'cov8']
        	columns of C for the covariates to adjust features by.
        adj_list: a csv file with header row
	        1st column (Feature) feature to adjust
        	2nd column (Type) adjustment type: 'logistic' or 'linear'
        	3rd column (Covariates) list of confounding covariates separated by ";"
        	e.g.
        	Feature,Type,Covariates
        	X1,logistic,cov4;cov7;cov8
        	X2,logistic,cov4
        	X6,linear,cov4
        	X15,linear,cov7;cov8
        	X20,logistic,cov4;cov8

        """

        self.C = C
        self.adj_list = adj_list

    def fit(self, X, y=None, **fit_params):
        """Fit the StackingEstimator meta-transformer.

        Parameters
        ----------
        X : array-like
        y: None. Ignored variable.
        fit_params: other estimator-specific parameters

        Returns
        -------
        self: object
        Returns an estimator for each feature that needs to be adjusted.

        """

        if self.C is None:
            raise ValueError('X is missing the covariate columns')
        if self.adj_list is None:
            raise ValueError('No adjustment information given')

        X_train = pd.DataFrame.copy(X)
        X_train.drop(self.C, axis=1, inplace=True)
        self.adj_df = pd.read_csv(self.adj_list)
        self.col_ests = {}
        for a in self.adj_df.Feature:
            if re.match(r'^indicator', a) or re.match(r'^adjY', a):
                raise ValueError("indicator and adjY columns of X should not be adjusted")
        self.comm_features = [a for a in self.adj_df.Feature if a in X_train.columns]
        if self.comm_features:
            self.sub_adj_df = self.adj_df[self.adj_df['Feature'].isin(self.comm_features)]
            for _, row in self.sub_adj_df.iterrows():
                subC = row['Covariates'].split(";")
                tmp_C = X[subC].values
                tmp_X = X_train[row['Feature']]
                est = X_adj_fit(tmp_X, tmp_C, row['Type'])
                self.col_ests[row['Feature']] = est

        return self

    def transform(self, X):
        """
        Parameters
        ----------
        X : array-like

        """

        if self.C is None:
            raise ValueError('X is missing the covariate columns')
        if self.adj_list is None:
            raise ValueError('No adjustment information given')

        X_test = pd.DataFrame.copy(X)
        X_test.drop(self.C, axis=1, inplace=True)
        X_test_adj = X_test.values
        self.adj_df = pd.read_csv(self.adj_list)

        if self.comm_features:
            # features not to adjust
            X_test_unsel = X_test.drop(self.comm_features, axis=1)

            # features to adjust
            X_subset_adj = np.array([]) # make a empty array

            for _, row in self.sub_adj_df.iterrows():
                subC = row['Covariates'].split(";")
                tmp_C = X[subC].values
                tmp_X = X_test[row['Feature']]
                tmp_X_adj = X_adj_predict(tmp_X, tmp_C, row['Type'],
                                          self.col_ests[row['Feature']])
                if X_subset_adj.size == 0:
                    X_subset_adj = tmp_X_adj
                else:
                    X_subset_adj = np.hstack((X_subset_adj, tmp_X_adj))

            X_subset_adj = pd.DataFrame(X_subset_adj, index=X_test_unsel.index)
            X_test_adj = pd.concat([X_subset_adj, X_test_unsel], axis=1)

        return X_test_adj
