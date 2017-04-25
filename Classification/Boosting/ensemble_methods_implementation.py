# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 13:44:18 2017

@author: Jihoon_Kim
"""

# Boosting a decision stump

# Decision Trees in Practice

# Import module
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from boosting_func import weighted_decision_tree_create
from boosting_func import evaluate_classification_error
from boosting_func import adaboost_with_tree_stumps
from boosting_func import predict_adaboost

# Load Data
loans = pd.read_csv("lending-club-data.csv")

# Explore target column
# safe_loans =  1 => safe
# safe_loans = -1 => risky
loans['safe_loans'] = loans['bad_loans'].apply(lambda x: 1 if x==0 else -1)
loans = loans.drop('bad_loans',1)

# Subset of features
features = ['grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home_ownership status: own, mortgage or rent
            'emp_length',         # number of years of employment
           ]
target = 'safe_loans'

# Extract the feature and target columns
loans = loans[features+[target]]

safe_loans_raw = loans[loans[target] == +1]
risky_loans_raw = loans[loans[target] == -1]

# Since there are fewer risky loans than safe loans, find the ratio of the sizes
# and use that percentage to undersample the safe loans.
percentage = len(risky_loans_raw)/(len(safe_loans_raw))
risky_loans = risky_loans_raw
safe_loans = safe_loans_raw.sample(frac=percentage)
# Append the risky_loans with the downsampled version of safe_loans
loans_data = risky_loans.append(safe_loans)

# one-hot encoding
print("Data types: \n", loans_data.dtypes)
categorical_variables = ['grade','term','home_ownership','emp_length']
loans_data = pd.get_dummies(loans_data,columns=categorical_variables)
train_data, test_data = train_test_split(loans_data,test_size=0.2)
features = list(loans_data.columns.values)
features.remove('safe_loans')
# Explore train_data
print(train_data.head(5))
# Explore test_data
print(test_data.head(5))

print("Number of Safe Loans in TRAIN DATA: ", (train_data['safe_loans']==1).sum())
print("Number of Bad Loans in TRAIN DATA: ", (train_data['safe_loans']==-1).sum())
print("Number of Safe Loans in TEST DATA: ", (test_data['safe_loans']==1).sum())
print("Number of Bad Loans in TEST DATA: ", (test_data['safe_loans']==-1).sum())

# Example: Training a weighted decision tree
example_data_weights = [1.] * 10 + [0.]*(len(train_data) - 20) + [1.] * 10
small_data_decision_tree_subset_20 = weighted_decision_tree_create(train_data, features, target,
                         example_data_weights, max_depth=2)

subset_20 = train_data.head(10).append(train_data.tail(10))
evaluate_classification_error(small_data_decision_tree_subset_20, subset_20, target)
evaluate_classification_error(small_data_decision_tree_subset_20, train_data, target)

# Implementing your own Adaboost (on decision stumps)
stump_weights, tree_stumps = adaboost_with_tree_stumps(train_data, features, 
                                target, num_tree_stumps=10)

# Making predictions
predictions = predict_adaboost(stump_weights, tree_stumps, test_data)
(test_data['safe_loans'] == predictions).sum()/len(test_data)

# Performance plots
stump_weights, tree_stumps = adaboost_with_tree_stumps(train_data, 
                                 features, target, num_tree_stumps=30)

# Computing training error at the end of each iteration
error_all = []
for n in range(1, 31):
    predictions = predict_adaboost(stump_weights[:n], tree_stumps[:n], train_data)
    error = 1.0 - (train_data[target] == predictions).sum()/len(train_data)
    error_all.append(error)
    print("Iteration %s, training error = %s" % (n, error_all[n-1]))
    

# Visualizing training error vs number of iterations
plt.figure()
plt.rcParams['figure.figsize'] = 7, 5
plt.plot(range(1,31), error_all, '-', linewidth=4.0, label='Training error')
plt.title('Performance of Adaboost ensemble')
plt.xlabel('# of iterations')
plt.ylabel('Classification error')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size': 16})

# Evaluation on the test data
test_error_all = []
for n in range(1, 31):
    predictions = predict_adaboost(stump_weights[:n], tree_stumps[:n], test_data)
    error = 1.0 - (test_data[target] == predictions).sum()/len(test_data)
    test_error_all.append(error)
    print("Iteration %s, test error = %s" % (n, test_error_all[n-1]))

# Visualize both the training and test errors
plt.figure()
plt.rcParams['figure.figsize'] = 7, 5
plt.plot(range(1,31), error_all, '-', linewidth=4.0, label='Training error')
plt.plot(range(1,31), test_error_all, '-', linewidth=4.0, label='Test error')

plt.title('Performance of Adaboost ensemble')
plt.xlabel('# of iterations')
plt.ylabel('Classification error')
plt.rcParams.update({'font.size': 16})
plt.legend(loc='best', prop={'size':15})
plt.tight_layout()