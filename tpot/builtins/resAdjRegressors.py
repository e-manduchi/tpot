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
Modification of regressors which handle indicator and adjY columns
when covariate adjustments are needed for the target.

"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
import re
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

class resAdjRegressor(BaseEstimator, RegressorMixin):
    """Meta-regressor for handling of  indicator and adjY columns."""

    def __init__(self, base_estimator, estimator_params=tuple()):
        """Create a resAdjRegressor object.

        Parameters
        ----------
        base_estimator: object with fit and predict methods.
            The base estimator to be modified so that it handles indicator and adjY columns.
        estimator_params = tuple
            The hyperparameters for the base estimator

        """

        self.base_estimator = estimator
        self.estimator_params = estimator_params

    def fit(self, X, y=None):
        """Set the hyperparamenters and fit the resAdjRegressor.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The training input samples.
        y: array-like, shape (n_samples,)
            The target values (this is actually adjY).
        
        Returns
        -------
        self: object
            The fitted regressor.

        """

        X_train = pd.DataFrame.copy(X)
        for col in X_train.columns:
            if re.match(r'^indicator', str(col)) or re.match(r'^adjY', str(col)):
                X_train.drop(col, axis=1, inplace=True)

        indX = X.filter(regex='indicator')
        if indX.shape[1] == 0:
            raise ValueError("X has no indicator columns")
        adjY = X.filter(regex='adjY')
        if (adjY.shape[1] == 0):
            raise ValueError("X has no adjY columns")

        y_train = y
        for col in indX.columns:
            if sum(indX[col])==0:
                i = col.split('_')[1]
                y_train = X['adjY_' + i]
                break

        estimator = clone(self.base_estimator)
        estimator.set_params(**{p: getattr(self, p)
                                for p in self.estimator_params})
        
        self.estimator =  estimator.fit(X_train, y_train)

        return self


    def predict(self, X):
        """Predicts targets.

        Parameters
        ----------
        X: numpy ndarray, {n_samples, n_components}

        Returns
        -------
        y_pred: array-like, shape (n_samples,)
            The predicted values.

        """

        X_test = pd.DataFrame.copy(X)
        for col in X_test.columns:
            if re.match(r'^indicator', str(col)) or re.match(r'^adjY', str(col)):
                X_test.drop(col, axis=1, inplace=True)

        return self.estimator.predict(X_test)

class resAdjAdaBoostRegressor(resAdjRegressor):
    """AdaBoostRegressor modified to handle indicator and adjY columns."""

    def __init__(self, n_estimators=100, learning_rate=1.,
                 loss='linear', random_state=None):
        super().__init__(base_estimator=AdaBoostRegressor(), estimator_params=("n_estimators",
                                                                               "learning_rate",
                                                                               "loss",
                                                                               "random_state"))

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.random_state = random_state

class resAdjDecisionTreeRegressor(resAdjRegressor):
    """DecisionTreeRegressor modified to handle indicator and adjY columns."""

    def __init__(self, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, random_state=None):
        super().__init__(base_estimator=DecisionTreeRegressor(), estimator_params=("max_depth", 
                                                                                   "min_samples_split",
                                                                                   "min_samples_leaf",
                                                                                   "random_state"))

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

class resAdjExtraTreesRegressor(resAdjRegressor):
    """ExtraTreesRegressor modified to handle indicator and adjY columns."""

    def __init__(self, n_estimators=100, max_features='auto', min_samples_split=2, 
                 min_samples_leaf=1, bootstrap=False, random_state=None):
        super().__init__(base_estimator=ExtraTreesRegressor(), estimator_params=("n_estimators", 
                                                                                 "max_features",
                                                                                 "min_samples_split",
                                                                                 "min_samples_leaf",
                                                                                 "bootstrap",
                                                                                 "random_state"))

        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.random_state = random_state

class resAdjGradientBoostingRegressor(resAdjRegressor):
    """GradientBoostingRegressor modified to handle indicator and adjY columns."""

    def __init__(self, n_estimators=100, loss='ls', learning_rate=0.1,
                 max_depth=3, min_samples_split=2, min_samples_leaf=1,
                 subsample=0.05, max_features=None, alpha=0.9, random_state=None):
        super().__init__(base_estimator=GradientBoostingRegressor(), estimator_params=("n_estimators", 
                                                                                       "loss",
                                                                                       "learning_rate",
                                                                                       "max_depth",
                                                                                       "min_samples_split",
                                                                                       "min_samples_leaf",
                                                                                       "subsample",
                                                                                       "max_features",
                                                                                       "alpha",
                                                                                       "random_state"))

       self.n_estimators = n_estimators 
       self.loss = loss
       self.learning_rate = learning_rate
       self.max_depth = max_depth
       self.min_samples_split = min_samples_split
       self.min_samples_leaf = min_samples_leaf
       self.subsample = subsample
       self.max_features = max_features
       self.alpha = alpha
       self.random_state = random_state

class resAdjKNeighborsRegressor(resAdjRegressor):
    """KNeighborsRegressor modified to handle indicator and adjY columns."""

    def __init__(self, n_neighbors=5, weights='uniform', p=2):
        super().__init__(base_estimator=KNeighborsRegressor(), estimator_params=("n_neighbors", 
                                                                                 "weights",
                                                                                 "p"))


        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
