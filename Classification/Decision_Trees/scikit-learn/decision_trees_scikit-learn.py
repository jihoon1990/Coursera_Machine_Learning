# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 13:10:12 2017

@author: Jihoon_Kim
"""
# Identifying safe loans with decision trees

# Import module
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from IPython.display import Image
from IPython.display import display
import pydotplus
# Load data
loans = pd.read_csv("lending-club-data.csv")

# Explore some features
print("Column Names: \n", loans.columns)

# Explore target column
# safe_loans =  1 => safe
# safe_loans = -1 => risky
loans['safe_loans'] = loans['bad_loans'].apply(lambda x: 1 if x==0 else -1)
loans = loans.drop('bad_loans',1)

# Let's explore the distribution of safeloans
print("Number of safe loans: ", sum(loans['safe_loans']==1))
print("Number of bad loans: ", sum(loans['safe_loans']==-1))
print("Total number of loans: ", len(loans))

# Subset of features
features = ['grade',                     # grade of the loan
            'sub_grade',                 # sub-grade of the loan
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'term',                      # the term of the loan
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
           ]

target = 'safe_loans'                    # prediction target (y) (+1 means safe, -1 is risky)

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

## Load index of train & validation
train_idx = pd.read_json("train-idx.json")
validation_idx = pd.read_json("validation-idx.json")
train_data = loans.iloc[train_idx[0].values]
validation_data = loans.iloc[validation_idx[0].values]

# one-hot encoding
print("Data types: \n", train_data.dtypes)
categorical_variables = ['grade','sub_grade','home_ownership','purpose','term']
train_one_hot_encoded = pd.get_dummies(train_data,columns=categorical_variables)
train_target = train_one_hot_encoded['safe_loans']
train_features = train_one_hot_encoded.drop('safe_loans',axis = 1)

# Build a decision tree classifier
decision_tree_model = DecisionTreeClassifier(max_depth=6)
decision_tree_model = decision_tree_model.fit(train_features,train_target)
small_model = DecisionTreeClassifier(max_depth=4)
small_model = small_model.fit(train_features,train_target)
big_model = DecisionTreeClassifier(max_depth=10)
big_model = big_model.fit(train_features,train_target)

# Visualize Tree
import sys
import os
def conda_fix(graph):
        path = os.path.join(sys.base_exec_prefix, "Library", "bin", "graphviz")
        paths = ("dot", "twopi", "neato", "circo", "fdp")
        paths = {p: os.path.join(path, "{}.exe".format(p)) for p in paths}
        graph.set_graphviz_executables(paths) 

dot_data = export_graphviz(small_model,out_file=None,feature_names= train_features.columns.values, class_names=['Bad','Safe'],filled=True,rounded=True,special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
conda_fix(graph)
display(Image(graph.create_png()))

# Making predictions
validation_safe_loans = validation_data[validation_data[target] == 1]
validation_risky_loans = validation_data[validation_data[target] == -1]
sample_validation_data_risky = validation_risky_loans[0:2]
sample_validation_data_safe = validation_safe_loans[0:2]
sample_validation_data = sample_validation_data_safe.append(sample_validation_data_risky)
# one-hot encoding
validation_one_encoded = pd.get_dummies(validation_data,columns=categorical_variables)
validation_features = validation_one_encoded.drop('safe_loans',axis=1)

# input
sample_validation_features = validation_features.loc[sample_validation_data.index]
# prediction
print("Predictions with max_depth = 6 (Class): ", decision_tree_model.predict(sample_validation_features))
decision_prob = pd.DataFrame(decision_tree_model.predict_proba(sample_validation_features))
decision_prob = decision_prob.rename(columns={0:'RISKY',1:'SAFE'})
print("Predictions with max_depth = 6 (Prob): \n", decision_prob)

print("Predictions with max_depth = 4 (Class): ", small_model.predict(sample_validation_features))
small_prob = pd.DataFrame(small_model.predict_proba(sample_validation_features))
small_prob = small_prob.rename(columns={0:'RISKY',1:'SAFE'})
print("Predictions with max_depth = 4 (Prob): \n", small_prob)

# Accuracy
print("Accuracy (Depth: 6, TRAIN DATA): %.4g" %decision_tree_model.score(train_features,train_data['safe_loans']))
print("Accuracy (Depth: 4, TRAIN DATA): %.4g" %small_model.score(train_features,train_data['safe_loans']))

print("Accuracy (Depth: 6, VALIDATION DATA): %.4g" %decision_tree_model.score(validation_features,validation_data['safe_loans']))
print("Accuracy (Depth: 4, VALIDATION DATA): %.4g" %small_model.score(validation_features,validation_data['safe_loans']))

print("=====TREE MODEL WITH DEPTH OF 10=====")
print("Accuracy (Depth: 10, TRAIN DATA): %.4g" %big_model.score(train_features,train_data['safe_loans']))
print("Accuracy (Depth: 10, VALIDATION DATA): %.4g" %big_model.score(validation_features,validation_data['safe_loans']))

# Quantifying the cost of mistakes

predictions = decision_tree_model.predict(validation_features)
dot_data = export_graphviz(decision_tree_model,out_file=None,feature_names= validation_features.columns.values, class_names=['Bad','Safe'],filled=True,rounded=True,special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
conda_fix(graph)
display(Image(graph.create_png()))

print("Number of Samples: ", len(predictions))
false_positives = (validation_data[validation_data['safe_loans'] != predictions]['safe_loans'] == -1).sum()
false_negatives = (validation_data[validation_data['safe_loans'] != predictions]['safe_loans'] == +1).sum()
print("FALSE Positive: ", false_positives)
print("FALSE Negative: ", false_negatives)

# Assume a cost of \$10,000 per false negative.
# Assume a cost of \$20,000 per false positive.
cost_of_mistakes = (false_negatives * 10000) + (false_positives * 20000)
print("Cost of Mistakes: ", cost_of_mistakes)