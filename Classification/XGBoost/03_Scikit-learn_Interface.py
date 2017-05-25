# -*- coding: utf-8 -*-
"""
Scikit-learn Interface
Created on Thu May 25 12:37:24 2017

@author: Jihoon_Kim
@contact: jioon_kim@outlook.com
"""

import numpy as np

from sklearn.datasets import load_svmlight_files
from sklearn.metrics import accuracy_score

from xgboost.sklearn import XGBClassifier

# Load Data

X_train, y_train, X_test, y_test = load_svmlight_files(
        ('./data/agaricus.txt.train', './data/agaricus.txt.test'))
print("Train dataset contains {0} rows and {1} columns".format(
        X_train.shape[0], X_train.shape[1]))
print("Test dataset contains {0} rows and {1} columns".format(
        X_test.shape[0], X_test.shape[1]))

print("Train possible labels: ")
print(np.unique(y_train))

print("\nTest possible labels: ")
print(np.unique(y_test))

# Train

params = {
    'objective': 'binary:logistic',
    'max_depth': 2,
    'learning_rate': 1.0,
    'silent': 1.0,
    'n_estimators': 5,
    }

bst = XGBClassifier(**params).fit(X_train, y_train)

# Predict
preds = bst.predict(X_test)
preds

correct = 0

for i in range(len(preds)):
    if (y_test[i] == preds[i]):
        correct += 1

acc = accuracy_score(y_test, preds)

print('Predicted correctly: {0}/{1}'.format(correct, len(preds)))
print('Error: {0:.4f}'.format(1-acc))
