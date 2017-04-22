# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 10:49:41 2017

@author: Jihoon_Kim
"""
# Implementing Binary Decision Trees

# Import module
import pandas as pd
from decision_trees_func import decision_tree_create
from decision_trees_func import classify
from decision_trees_func import evaluate_classification_error

# Load data
loans = pd.read_csv("lending-club-data.csv")

# Explore some features
print("Column Names: \n", loans.columns)

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

## Load train & test
train_idx = pd.read_json("train-idx.json")
test_idx = pd.read_json("test-idx.json")
train_data = loans.iloc[train_idx[0].values]
test_data = loans.iloc[test_idx[0].values]

# one-hot encoding
print("Data types: \n", train_data.dtypes)
categorical_variables = ['grade','term','home_ownership','emp_length']
train_data = pd.get_dummies(train_data,columns=categorical_variables)
train_target = train_data['safe_loans']
train_features = train_data.drop('safe_loans',axis = 1)
features = list(train_features.columns.values)
print("Number of Features (after one-hot encoding): ", len(train_features.columns))

test_data = pd.get_dummies(test_data,columns=categorical_variables)
test_target = test_data['safe_loans']
test_features = test_data.drop('safe_loans', axis = 1)

# Explore one-hot-encoded value
print("One-hot encoded (TRAIN DATA): \n", train_data.head(5))

# Build Tree
# Make sure to cap the depth at 6 by using max_depth = 6
my_decision_tree = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6)
print(my_decision_tree)

# Let's predict class for test_data[0]
print()
print("Test Data [0]: \n", test_data.iloc[0])
print()
print("Predicted class for test_data[0] (my_decision_tree): ", classify(my_decision_tree,test_data.iloc[0],annotate=True))

# Evaluating your decision tree
print("Error of Test Data: ", evaluate_classification_error(my_decision_tree, test_data, target))