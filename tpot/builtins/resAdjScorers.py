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
Implementation of (regression) scorers to be used when covariate adjustments 
are needed for the target and it is expected that X contains indicator and
adjY columns as generated by the resAdjTpotPreprocessor script.

"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error

def getYtrue(X, y):
    """Return the adjY column corresponding to the CV split of interest.

    Parameters
    ----------
    X : array-like or sparse matrix
    	Test data that will be fed to estimator.predict.
    	Must contain indicator and adjY columns.
    y : array-like
    	This is adjY.

    """

    indX = X.filter(regex='indicator')
    if indX.shape[1] == 0:
        raise ValueError("X has no indicator columns")
    adjY = X.filter(regex='adjY')
    if (adjY.shape[1] == 0):
        raise ValueError("X has no adjY columns")
    y_true = y
    for col in indX.columns:
        if sum(indX[col])==indX.shape[0]:
            i = col.split('_')[1]
            y_true = X['adjY_' + i]
            break
    return y_true

def resAdjMseScorer(estimator, X, y):
    """Compute neg_mean_squared_error for predictions on X and y_true.

    Parameters
    ----------
    X : array-like or sparse matrix
    	Test data that will be fed to estimator.predict.
    	Must contain indicator and adjY columns.
    y : array-like
    	This is adjY.

    Returns
    -------
    score : float
    	neg_mean_squared_error.

    """

    y_pred = estimator.predict(X)
    y_true = getYtrue(X, y)
    if len(y_true) != len(y_pred):
        raise ValueError("y_pred and y_true have different lengths")

    score = -(mean_squared_error(y_true, y_pred))
    return score

def resAdjR2Scorer(estimator, X, y):
    """Compute r2_score for predictions on X and y_true.

    Parameters
    ----------
    X : array-like or sparse matrix
    	Test data that will be fed to estimator.predict.
    	Must contain indicator and adjY columns.
    y : array-like
   	This is adjY.

    Returns
    -------
    score : float
    	r2_score.

    """

    y_pred = estimator.predict(X)
    y_true = getYtrue(X, y)
    if len(y_true) != len(y_pred):
        raise ValueError("y_pred and y_true have different lengths")

    score = r2_score(y_true, y_pred)
    return score

def resAdjMaeScorer(estimator, X, y):
    """Compute neg_mean_absolute error for predictions on X and y_true.

    Parameters
    ----------
    X : array-like or sparse matrix
    	Test data that will be fed to estimator.predict.
    	Must contain indicator and adjY columns.
    y : array-like
    	This is adjY.

    Returns
    -------
    score : float
    	neg_mean_absolute_error.

    """

    y_pred = estimator.predict(X)
    y_true = getYtrue(X, y)
    if len(y_true) != len(y_pred):
        raise ValueError("y_pred and y_true have different lengths")

    score = -(mean_absolute_error(y_true, y_pred))
    return score

def resAdjMedAeScorer(estimator, X, y):
    """Compute neg_median_absolute_error for predictions on X and y_true.

    Parameters
    ----------
    X : array-like or sparse matrix
	    Test data that will be fed to estimator.predict.
    	Must contain indicator and adjY columns.
    y : array-like
    	This is adjY.

    Returns
    -------
    score : float
    	neg_median_absolute_error.

    """

    y_pred = estimator.predict(X)
    y_true = getYtrue(X, y)
    if len(y_true) != len(y_pred):
        raise ValueError("y_pred and y_true have different lengths")

    score = -(median_absolute_error(y_true, y_pred))
    return score
