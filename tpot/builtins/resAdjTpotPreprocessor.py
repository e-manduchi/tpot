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
Pre-processing script for TPOT with covariate adjustment on target.

Description
-----------
This script prepares the training input for TPOT when covariate
adjustments are needed for the target. The input to this script
should contain the feature columns X, the target column y, and 
the covariate columns C. The columns of C corresponds to covariates
needed to adjust some (or all) of the X columns and/or the y column.
If no adjustments are needed for y, this script should not be used.
If adjustments are  needed for y by some of the columns in C, this
script should be used and any of these columns that is no longer needed
for subsequent feature adjustments should be specified so that
it is removed from the script output.
This script adds to the input N additional column pairs, where N is the
number of CV splits one plans to use in TPOT. Each such pair consists of
an indicator column, denoting  training and testing rows for that split, 
and a column holding the adjusted target (via residuals). The adjustment 
uses a linear or logistic regression fitted on the training subset only.
In addition, this script replaces the original target column y by its 
adjustment using the entire training set. Finally, this script removes 
the covariate columns that are not needed for subsequent feature adjustments.
The script can also take an optional held-out testing file and preprocesses it
so that it can be used to evaluate the resAdj TPOT fitted pipeline. If this 
testing file is provided, it is assumed that it has the same names for the 
index, target, and covariate columns as the training input file.

Arguments
---------
1. Input tab-delimited training file.
2. Optional tab-delimited held-out testing file (file path or None).
3. Column name for the index column (holding sample ids) if present.
4. Column name for the target variable y.
5. One of 'regression' or 'classification', depending on y.
6. Number of CV splits to generate.
7. Output training file.
8. Output held-out testing file (file path or None).
9. Comma-separated list (no spaces) of the covariate columns to use to adjust y.
10. Optional comma-separated list (no spaces) of covariate columns to remove from
the output, as not needed for feature adjustments.

Example usage
-------------
python resAdjTpotPreprocessor.py inputTrain.txt None ids class classification 5 outputTrain.txt None cov1,cov2,cov3 cov1,cov3

"""

import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

if len(sys.argv)<10:
        sys.exit("Insufficient number of arguments passed in")

inTrainFile = sys.argv[1]
inTestFile = sys.argv[2]
indexCol = sys.argv[3]
targetCol = sys.argv[4]
mode = sys.argv[5]
numSplits = int(sys.argv[6])
outTrainFile = sys.argv[7]
outTestFile = sys.argv[8]
cov = sys.argv[9].split(',')

if (inTestFile is None and outTestFile is not None) or (inTestFile is not None and outTestFile is None):
        raise ValueError("If one of the names for input Test file or output Test file is provided, the other has to be provided too")

if (mode!='regression' and mode!='classification'):
        sys.exit("mode must be one of 'regression' or 'classification'")

def getPredictions(estimator, B):
        """Get estimator predictions.

        Parameters
        ----------
        estimator: object
        	The fitted estimator to use for predictions.
        B: array-like of shape (n_samples, n_covariates)
        	The covariate values for target adjustment.

        Returns
        -------
        pi: array-like of shape (n_samples,)
        	Predicted value per samples. For classification mode, this is the predicted expected class value.

        """

        if mode == 'classification':
            predProba = estimator.predict_proba(B)
            pi = np.zeros((predProba.shape[0], ))
            for idx, gt in enumerate(estimator.classes_):
                gt = int(gt)
                pi = pi + gt*predProba[:, idx]
        else:
            pi = estimator.predict(B)
        return pi

data = pd.read_csv(inTrainFile, sep='\t', index_col=indexCol)
Xdata = data.drop(targetCol, axis=1)
Ydata = data[targetCol]
B = data[cov]

if inTestFile != 'None':
        testData = pd.read_csv(inTestFile, sep='\t', index_col=indexCol)
        Ydata_test = testData[targetCol]
        B_test = testData[cov]

for i in range(numSplits+1):
        if mode == 'classification':
                estimator = LogisticRegression(penalty='none',
                                               solver='lbfgs',
                                               multi_class='auto',
                                               max_iter=500)
        else:
                estimator = LinearRegression()

        if i==0:
                estimator.fit(B, Ydata)
                Yadj = pd.Series(index=data.index, dtype=float)
                Yadj = Ydata - getPredictions(estimator, B)
                Yadj.rename('adjY', inplace=True)
                data = data.join(Yadj)

                if inTestFile != 'None':
                        Yadj_test = pd.Series(index=testData.index, dtype=float)
                        Yadj_test = Ydata_test - getPredictions(estimator, B_test)
                        Yadj_test.rename('adjY', inplace=True)
                        testData = testData.join(Yadj_test)
                        oneData = np.ones(len(Ydata_test), dtype=int)
                        indicator1 = pd.Series(oneData, index=testData.index, name='indicator_1')
                        adjY1 = Yadj_test.copy()
                        adjY1.rename('adjY_1', inplace=True)
                        testData = testData.join([indicator1, adjY1])  # Mock placeholder columns needed for the pipeline to run on the testing set
        else:
                seed = 42 + i
                if mode == 'regression':
                        Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xdata, Ydata, random_state=int(seed), train_size=0.75, test_size=0.25)
                else:
                        Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xdata, Ydata, random_state=int(seed), train_size=0.75, test_size=0.25, stratify=Ydata)


                zeroData = np.zeros(len(Ydata), dtype=int)
                indicator = pd.Series(zeroData, index=data.index, name='indicator_' + str(i))
                indicator[Xtest.index] = 1

                Btrain = B.loc[Xtrain.index]
                Btest = B.loc[Xtest.index]

                estimator.fit(Btrain, Ytrain)
                Yadj = pd.Series(index=data.index, dtype=float, name='adjY_' + str(i))

                Yadj[Xtrain.index] = Ytrain - getPredictions(estimator, Btrain)
                Yadj[Xtest.index] = Ytest - getPredictions(estimator, Btest)

                data = data.join(indicator)
                data = data.join(Yadj)

data.drop(targetCol, axis=1, inplace=True)
if inTestFile != 'None':
        testData.drop(targetCol, axis=1, inplace=True)

if len(sys.argv)>10:
        remove = sys.argv[10].split(',')
        data.drop(remove, axis=1, inplace=True)
        if inTestFile != 'None':
                testData.drop(remove, axis=1, inplace=True)

data.to_csv(outTrainFile, sep='\t', header = True)
if inTestFile != 'None':
        testData.to_csv(outTestFile, sep='\t', header=True)
