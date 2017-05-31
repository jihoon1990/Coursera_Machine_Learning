# -*- coding: utf-8 -*-

from collections import Counter

from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss, accuracy_score

# classifiers
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

# reproducibility
seed = 104

"""
Prepare Data
"""
X, y = make_classification(n_samples=1000, n_features=20, n_informative=8,
                           n_redundant=3, n_repeated=2, random_state=seed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=seed)

print("Train label distribution:")
print(Counter(y_train))
print("\nTest label distribution:")
print(Counter(y_test))

"""
Single Decision Tree
"""

decision_tree = DecisionTreeClassifier(random_state=seed)

# train classifier
decision_tree.fit(X_train, y_train)

# predict output
decision_tree_y_pred = decision_tree.predict(X_test)
decision_tree_y_pred_prob = decision_tree.predict_proba(X_test)

# evaluation
decision_tree_accuracy = accuracy_score(y_test, decision_tree_y_pred)
decision_tree_logloss = log_loss(y_test, decision_tree_y_pred_prob)

print("== Decision Tree ==")
print("Accuracy: {0:.2f}".format(decision_tree_accuracy))
print("Log loss: {0:.2f}".format(decision_tree_logloss))
print("Number of nodes created: {}".format(decision_tree.tree_.node_count))

print('True labels:')
print(y_test[:5, ])
print('\nPredicted labels:')
print(decision_tree_y_pred[:5, ])
print('\nPredicted probabilities:')
print(decision_tree_y_pred_prob[:5, ])

dt_viz_file = '.\\images\\dt.dot'

# create visualization
export_graphviz(decision_tree, out_file=dt_viz_file)

"""
AdaBoost
"""
adaboost = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    algorithm='SAMME',
    n_estimators=1000,
    random_state=seed)

# train classifier
adaboost.fit(X_train, y_train)

# calculate predictions
adaboost_y_pred = adaboost.predict(X_test)
adaboost_y_pred_prob = adaboost.predict_proba(X_test)

# evaluate
adaboost_accuracy = accuracy_score(y_test, adaboost_y_pred)
adaboost_logloss = log_loss(y_test, adaboost_y_pred_prob)

print("== AdaBoost ==")
print("Accuracy: {0:.2f}".format(adaboost_accuracy))
print("Log loss: {0:.2f}".format(adaboost_logloss))

print('True labels:')
print(y_test[:5, ])
print('\nPredicted labels:')
print(adaboost_y_pred[:5, ])
print('\nPredicted probabilities:')
print(adaboost_y_pred_prob[:5, ])


ada_t1 = adaboost.estimators_[0]
ada_t1_viz_file = '.\\images\\ada-t1.dot'
# create visualization
export_graphviz(ada_t1, out_file=ada_t1_viz_file)

print("Error: {0:.2f}".format(adaboost.estimator_errors_[0]))
print("Tree importance: {0:.2f}".format(adaboost.estimator_weights_[0]))

"""
Gradient Boosted Trees
"""
gbc = GradientBoostingClassifier(
    max_depth=1,
    n_estimators=1000,
    warm_start=True,
    random_state=seed)
gbc.fit(X_train, y_train)

# make predictions
gbc_y_pred = gbc.predict(X_test)
gbc_y_pred_prob = gbc.predict_proba(X_test)

# calculate log loss
gbc_accuracy = accuracy_score(y_test, gbc_y_pred)
gbc_logloss = log_loss(y_test, gbc_y_pred_prob)

print("== Gradient Boosting ==")
print("Accuracy: {0:.2f}".format(gbc_accuracy))
print("Log loss: {0:.2f}".format(gbc_logloss))

print('True labels:')
print(y_test[:5, ])
print('\nPredicted labels:')
print(gbc_y_pred[:5, ])
print('\nPredicted probabilities:')
print(gbc_y_pred_prob[:5, ])

gbc_t1 = gbc.estimators_[2][0]
gbc_t1_viz_file = '.\\images\\gbc-t1.dot'

# create visualization
export_graphviz(gbc_t1, out_file=gbc_t1_viz_file)
