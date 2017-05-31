# -*- coding: utf-8 -*-

import numpy as np
import xgboost as xgb

dtrain = xgb.DMatrix('./data/agaricus.txt.train')
dtest = xgb.DMatrix('./data/agaricus.txt.test')

print("Train dataset contains {0} rows and {1} columns".format(
        dtrain.num_row(), dtrain.num_col()))
print("Test dataset contains {0} rows and {1} columns".format(
        dtest.num_row(), dtest.num_col()))

print("Train possible labels: ")
print(np.unique(dtrain.get_label()))

print("\nTest possible labels: ")
print(np.unique(dtest.get_label()))

params = {
    'objective': 'binary:logistic',
    'max_depth': 2,
    'silent': 1,
    'eta': 1
}

num_rounds = 5

# train
bst = xgb.train(params, dtrain, num_rounds)

watchlist = [(dtest, 'test'), (dtrain, 'train')]  # native interface only
bst = xgb.train(params, dtrain, num_rounds, watchlist)

# Predictions
preds_prob = bst.predict(dtest)
preds_prob

labels = dtest.get_label()
preds = preds_prob > 0.5  # threshold
correct = 0

for i in range(len(preds)):
    if (labels[i] == preds[i]):
        correct += 1

print('Predicted correctly: {0}/{1}'.format(correct, len(preds)))
print('Error: {0:.4f}'.format(1-correct/len(preds)))
