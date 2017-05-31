# -*- coding: utf-8 -*-

# Bias/Variance Trade-off

import numpy as np
import matplotlib.pyplot as plt
from sklearn.learning_curve import validation_curve
from sklearn.cross_validation import StratifiedKFold
from sklearn.datasets import make_classification
from xgboost.sklearn import XGBClassifier

# reproducibility
seed = 123
np.random.seed(seed)

X, y = make_classification(n_samples=1000, n_features=20, n_informative=8,
                           n_redundant=3, n_repeated=2, random_state=seed)

# 10 Fold
cv = StratifiedKFold(y, n_folds=10, shuffle=True, random_state=seed)

default_params = {
    'objective': 'binary:logistic',
    'max_depth': 1,
    'learning_rate': 0.3,
    'silent': 1.0
}

n_estimators_range = np.linspace(1, 200, 10).astype('int')

train_scores, test_scores = validation_curve(
    XGBClassifier(**default_params),
    X, y,
    param_name='n_estimators',
    param_range=n_estimators_range,
    cv=cv,
    scoring='accuracy'
)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

fig = plt.figure(figsize=(10, 6), dpi=100)

plt.title("Validation Curve with XGBoost (eta = 0.3)")
plt.xlabel("number of trees")
plt.ylabel("Accuracy")
plt.ylim(0.7, 1.1)

plt.plot(n_estimators_range,
         train_scores_mean,
         label="Training score",
         color="r")

plt.plot(n_estimators_range,
         test_scores_mean,
         label="Cross-validation score",
         color="g")

plt.fill_between(n_estimators_range,
                 train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std,
                 alpha=0.2, color="r")

plt.fill_between(n_estimators_range,
                 test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std,
                 alpha=0.2, color="g")

plt.axhline(y=1, color='k', ls='dashed')

plt.legend(loc="best")
plt.show()

i = np.argmax(test_scores_mean)
print("Best cross-validation result ({0:.2f}) obtained for {1}\
      trees".format(test_scores_mean[i], n_estimators_range[i]))

default_params = {
    'objective': 'binary:logistic',
    'max_depth': 2,  # changed
    'learning_rate': 0.3,
    'silent': 1.0,
    'colsample_bytree': 0.6,  # added
    'subsample': 0.7  # added
}

n_estimators_range = np.linspace(1, 200, 10).astype('int')

train_scores, test_scores = validation_curve(
    XGBClassifier(**default_params),
    X, y,
    param_name='n_estimators',
    param_range=n_estimators_range,
    cv=cv,
    scoring='accuracy'
)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

fig = plt.figure(figsize=(10, 6), dpi=100)

plt.title("Validation Curve with XGBoost (eta = 0.3)")
plt.xlabel("number of trees")
plt.ylabel("Accuracy")
plt.ylim(0.7, 1.1)

plt.plot(n_estimators_range,
         train_scores_mean,
         label="Training score",
         color="r")

plt.plot(n_estimators_range,
         test_scores_mean,
         label="Cross-validation score",
         color="g")

plt.fill_between(n_estimators_range,
                 train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std,
                 alpha=0.2, color="r")

plt.fill_between(n_estimators_range,
                 test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std,
                 alpha=0.2, color="g")

plt.axhline(y=1, color='k', ls='dashed')

plt.legend(loc="best")
plt.show()

i = np.argmax(test_scores_mean)
print("Best cross-validation result ({0:.2f}) obtained for {1}\
      trees".format(test_scores_mean[i], n_estimators_range[i]))
