# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 16:31:07 2017

@author: Jihoon_Kim
"""

# Decision Trees in Practice

# Import module
import pandas as pd
from decision_trees_func import decision_tree_create
from decision_trees_func import classify
from decision_trees_func import evaluate_classification_error
from decision_trees_func import count_leaves

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

# Subsample dataset to make sure classes are balanced
safe_loans_raw = loans[loans[target] == +1]
risky_loans_raw = loans[loans[target] == -1]

# Since there are fewer risky loans than safe loans, find the ratio of the sizes
# and use that percentage to undersample the safe loans.
percentage = len(risky_loans_raw)/(len(safe_loans_raw))

risky_loans = risky_loans_raw
safe_loans = safe_loans_raw.sample(frac=percentage)

# Append the risky_loans with the downsampled version of safe_loans
loans_data = risky_loans.append(safe_loans)

# Load train & test data
train_idx = pd.read_json("train-idx.json")
validation_idx = pd.read_json("validation-idx.json")
train_data = loans.iloc[train_idx[0].values]
validation_data = loans.iloc[validation_idx[0].values]

# Transform categorical data into binary features

# one-hot encoding
print("Data types: \n", train_data.dtypes)
categorical_variables = ['grade','term','home_ownership','emp_length']
train_data = pd.get_dummies(train_data,columns=categorical_variables)
train_target = train_data['safe_loans']
train_features = train_data.drop('safe_loans',axis = 1)
features = list(train_features.columns.values)
print("Number of Features (after one-hot encoding): ", len(train_features.columns))

validation_data = pd.get_dummies(validation_data,columns=categorical_variables)
validation_target = validation_data['safe_loans']
validation_features = validation_data.drop('safe_loans', axis = 1)

# Build a Tree
print("==================================")
print("MY_DECISION_TREE_NEW")
my_decision_tree_new = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, 
                                min_node_size = 100, min_error_reduction=0.0)
print("==================================")
print("MY_DECISION_TREE_OLD")
my_decision_tree_old = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, 
                                min_node_size = 0, min_error_reduction=-1)

# Predictions
# Let's predict class for validation_data[0]
print()
print("Test Data [0]: \n", validation_data.iloc[0])
print()
print("Predicted class for validation_data[0] (my_decision_tree_new): ", classify(my_decision_tree_new,validation_data.iloc[0],annotate=True))
print("Predicted class for validation_data[0] (my_decision_tree_old): ", classify(my_decision_tree_old,validation_data.iloc[0],annotate=True))

# Evaluating your decision tree
print("Error of Validation Data (New Tree): %.4g" %evaluate_classification_error(my_decision_tree_new, validation_data, target))
print("Error of Validaiton Data (Old Tree: %.4g" %evaluate_classification_error(my_decision_tree_old, validation_data, target))

# Exploring the effect of max_depth
print("==========EFFECT OF MAX_DEPTH==========")
model_1 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 2, 
                                min_node_size = 0, min_error_reduction=-1)
model_2 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, 
                                min_node_size = 0, min_error_reduction=-1)
model_3 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 14, 
                                min_node_size = 0, min_error_reduction=-1)

print("==========ERROR TABLE==========")
print("Training data, classification error (model 1):", evaluate_classification_error(model_1, train_data, target))
print("Training data, classification error (model 2):", evaluate_classification_error(model_2, train_data, target))
print("Training data, classification error (model 3):", evaluate_classification_error(model_3, train_data, target))

print("validation_set, classification error (model 1):", evaluate_classification_error(model_1, validation_data, target))
print("validation_set, classification error (model 2):", evaluate_classification_error(model_2, validation_data, target))
print("validation_set, classification error (model 3):", evaluate_classification_error(model_3, validation_data, target))

print("Number of nodes (model 1):", count_leaves(model_1))
print("Number of nodes (model 2):", count_leaves(model_2))
print("Number of nodes (model 3):", count_leaves(model_3))

# Exploring the effect of min_error
print("==========EFFECT OF MIN_ERROR==========")
model_4 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, 
                                min_node_size = 0, min_error_reduction=-1)
model_5 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, 
                                min_node_size = 0, min_error_reduction=0)
model_6 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, 
                                min_node_size = 0, min_error_reduction=5)

print("==========ERROR TABLE==========")
print("Training data, classification error (model 4):", evaluate_classification_error(model_4, train_data, target))
print("Training data, classification error (model 5):", evaluate_classification_error(model_5, train_data, target))
print("Training data, classification error (model 6):", evaluate_classification_error(model_6, train_data, target))

print("Validation data, classification error (model 4):", evaluate_classification_error(model_4, validation_data, target))
print("Validation data, classification error (model 5):", evaluate_classification_error(model_5, validation_data, target))
print("Validation data, classification error (model 6):", evaluate_classification_error(model_6, validation_data, target))

print("Number of nodes (model 4):", count_leaves(model_4))
print("Number of nodes (model 5):", count_leaves(model_5))
print("Number of nodes (model 6):", count_leaves(model_6))

# Exploring the effect of min_node_size
print("==========EFFECT OF MIN_NODE_SIZE==========")

model_7 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, 
                                min_node_size = 0, min_error_reduction=-1)
model_8 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, 
                                min_node_size = 2000, min_error_reduction=-1)
model_9 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, 
                                min_node_size = 50000, min_error_reduction=-1)

print("==========ERROR TABLE==========")
print("Training data, classification error (model 7):", evaluate_classification_error(model_7, train_data, target))
print("Training data, classification error (model 8):", evaluate_classification_error(model_8, train_data, target))
print("Training data, classification error (model 9):", evaluate_classification_error(model_9, train_data, target))

print("Validation data, classification error (model 7):", evaluate_classification_error(model_7, validation_data, target))
print("Validation data, classification error (model 8):", evaluate_classification_error(model_8, validation_data, target))
print("Validation data, classification error (model 9):", evaluate_classification_error(model_9, validation_data, target))

print("Number of nodes (model 7):", count_leaves(model_7))
print("Number of nodes (model 8):", count_leaves(model_8))
print("Number of nodes (model 9):", count_leaves(model_9))