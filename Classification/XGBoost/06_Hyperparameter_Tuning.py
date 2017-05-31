# -*- coding: utf-8 -*-

import numpy as np

from xgboost.sklearn import XGBClassifier

from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import make_classification
from sklearn.cross_validation import StratifiedKFold

from scipy.stats import randint, uniform

# reproducibility
seed = 342
np.random.seed(seed)

X, y = make_classification(n_samples=1000, n_features=20,
                           n_informative=8, n_redundant=3, n_repeated=2,
                           random_state=seed)
cv = StratifiedKFold(y, n_folds=10, shuffle=True, random_state=seed)

# Grid Search
params_grid = {
    'max_depth': [1, 2, 3],
    'n_estimators': [5, 10, 25, 50],
    'learning_rate': np.linspace(1e-16, 1, 3)
}

params_fixed = {
    'objective': 'binary:logistic',
    'silent': 1
}

bst_grid = GridSearchCV(
    estimator=XGBClassifier(**params_fixed, seed=seed),
    param_grid=params_grid,
    cv=cv,
    scoring='accuracy'
)

bst_grid.fit(X, y)
print(bst_grid.grid_scores_)
print("Best accuracy obtained: {0}".format(bst_grid.best_score_))
print("Parameters:")
for key, value in bst_grid.best_params_.items():
    print("\t{}: {}".format(key, value))


# Randomized Grid-Search
params_dist_grid = {
    'max_depth': [1, 2, 3, 4],
    'gamma': [0, 0.5, 1],
    'n_estimators': randint(1, 1001),  # uniform discrete random distribution
    'learning_rate': uniform(),  # gaussian distribution
    'subsample': uniform(),  # gaussian distribution
    'colsample_bytree': uniform()  # gaussian distribution
}


rs_grid = RandomizedSearchCV(
    estimator=XGBClassifier(**params_fixed, seed=seed),
    param_distributions=params_dist_grid,
    n_iter=10,
    cv=cv,
    scoring='accuracy',
    random_state=seed
)

rs_grid.fit(X, y)
print(rs_grid.grid_scores_)
print(rs_grid.best_estimator_)
print(rs_grid.best_params_)
print(rs_grid.best_score_)
