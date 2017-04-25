# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 11:21:37 2017

@author: Jihoon_Kim
"""

# Identifying safe loans with decision trees

# Import module
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

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

# Selecting Features
target = 'safe_loans'
features = ['grade',                     # grade of the loan (categorical)
            'sub_grade_num',             # sub-grade of the loan as a number from 0 to 1
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'payment_inc_ratio',         # ratio of the monthly payment to income
            'delinq_2yrs',               # number of delinquincies
             'delinq_2yrs_zero',          # no delinquincies in last 2 years
            'inq_last_6mths',            # number of creditor inquiries in last 6 months
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'open_acc',                  # number of open credit accounts
            'pub_rec',                   # number of derogatory public records
            'pub_rec_zero',              # no derogatory public records
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
            'int_rate',                  # interest rate of the loan
            'total_rec_int',             # interest received to date
            'annual_inc',                # annual income of borrower
            'funded_amnt',               # amount committed to the loan
            'funded_amnt_inv',           # amount committed by investors for the loan
            'installment',               # monthly payment owed by the borrower
           ]

# Skipping observations with missing values
loans = loans[[target] + features].dropna()

## Load index of train & validation
train_idx = pd.read_json("train-idx.json")
validation_idx = pd.read_json("validation-idx.json")
train_data = loans.iloc[train_idx[0].values]
validation_data = loans.iloc[validation_idx[0].values]

# one-hot encoding
print("Data types: \n", train_data.dtypes)
categorical_variables = ['grade','home_ownership','purpose']
train_one_hot_encoded = pd.get_dummies(train_data,columns=categorical_variables)
train_target = train_one_hot_encoded['safe_loans']
train_features = train_one_hot_encoded.drop('safe_loans',axis = 1)

# Build a classifier
ensemble_classifier = GradientBoostingClassifier(max_depth=6, n_estimators=5).fit(train_features,train_target)

# Making predictions
validation_safe_loans = validation_data[validation_data[target] == 1]
validation_risky_loans = validation_data[validation_data[target] == -1]
sample_validation_data_risky = validation_risky_loans[0:2]
sample_validation_data_safe = validation_safe_loans[0:2]
sample_validation_data = sample_validation_data_safe.append(sample_validation_data_risky)
# one-hot encoding for validation data
validation_one_encoded = pd.get_dummies(validation_data,columns=categorical_variables)
validation_features = validation_one_encoded.drop('safe_loans',axis=1)
# prediction
sample_validation_features = validation_features.loc[sample_validation_data.index]
print("Prediction with ensemble method (max_depth = 6, estimators: 5):, ", ensemble_classifier.predict(sample_validation_features))
ensemble_prob = pd.DataFrame(ensemble_classifier.predict_proba(sample_validation_features))
ensemble_prob = ensemble_prob.rename(columns={0:'RISKY',1:'SAFE'})
print("Prediction with ensemble method (max_depth = 6, estimators: 5) (Prob): \n", ensemble_prob)

# Accuracy
print("Accuracy (Validation Data): %.4g" %ensemble_classifier.score(validation_features,validation_data['safe_loans']))

# Quantifying the cost of mistakes
predictions = ensemble_classifier.predict(validation_features)
print("Number of Samples: ", len(predictions))
false_positives = (validation_data[validation_data['safe_loans'] != predictions]['safe_loans'] == -1).sum()
false_negatives = (validation_data[validation_data['safe_loans'] != predictions]['safe_loans'] == +1).sum()
print("FALSE Positive: ", false_positives)
print("FALSE Negative: ", false_negatives)

# Comparison with decision trees
# Assume a cost of \$10,000 per false negative.
# Assume a cost of \$20,000 per false positive.
cost_of_mistakes = (false_negatives * 10000) + (false_positives * 20000)
print("Cost of Mistakes: ", cost_of_mistakes)

# Most positive & negative loans.
prob = pd.DataFrame(ensemble_classifier.predict_proba(validation_features)[:,1],index=validation_features.index, columns=['predictions'])
validation_data = validation_data.assign(predictions = prob)
print("Most Positive Predictions: \n", validation_data[['grade','predictions']].sort_values(by='predictions',ascending = False).head(5))
print("Most Negative PRedictions: \n", validation_data[['grade','predictions']].sort_values(by='predictions',ascending = True).head(5))


# Effect of adding more trees
ensemble_classifier_est_10 = GradientBoostingClassifier(max_depth=6, n_estimators=10).fit(train_features,train_target)
ensemble_classifier_est_50 = GradientBoostingClassifier(max_depth=6, n_estimators=50).fit(train_features,train_target)
ensemble_classifier_est_100 = GradientBoostingClassifier(max_depth=6, n_estimators=100).fit(train_features,train_target)
ensemble_classifier_est_200 = GradientBoostingClassifier(max_depth=6, n_estimators=200).fit(train_features,train_target)
ensemble_classifier_est_500 = GradientBoostingClassifier(max_depth=6, n_estimators=500).fit(train_features,train_target)

# Compare accuracy on entire validation set
# Accuracy
print("Accuracy (n_estimator = 10, Validation Data): %.4g" %ensemble_classifier_est_10.score(validation_features,validation_data['safe_loans']))
print("Accuracy (n_estimator = 50, Validation Data): %.4g" %ensemble_classifier_est_50.score(validation_features,validation_data['safe_loans']))
print("Accuracy (n_estimator = 100, Validation Data): %.4g" %ensemble_classifier_est_100.score(validation_features,validation_data['safe_loans']))
print("Accuracy (n_estimator = 200, Validation Data): %.4g" %ensemble_classifier_est_200.score(validation_features,validation_data['safe_loans']))
print("Accuracy (n_estimator = 500, Validation Data): %.4g" %ensemble_classifier_est_500.score(validation_features,validation_data['safe_loans']))

# Plot the training and validation error vs. number of trees
def make_figure(dim, title, xlabel, ylabel, legend):
    plt.rcParams['figure.figsize'] = dim
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend is not None:
        plt.legend(loc=legend, prop={'size':15})
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()

# Training Errors
train_err_10 = 1 - ensemble_classifier_est_10.score(train_features,train_data['safe_loans'])
train_err_50 = 1 - ensemble_classifier_est_50.score(train_features,train_data['safe_loans'])
train_err_100 = 1 - ensemble_classifier_est_100.score(train_features,train_data['safe_loans'])
train_err_200 = 1 - ensemble_classifier_est_200.score(train_features,train_data['safe_loans'])
train_err_500 = 1 - ensemble_classifier_est_500.score(train_features,train_data['safe_loans'])
training_errors = [train_err_10, train_err_50, train_err_100, train_err_200, train_err_500]

# Validation Errors
validation_err_10 = 1 - ensemble_classifier_est_10.score(validation_features,validation_data['safe_loans'])
validation_err_50 = 1 - ensemble_classifier_est_50.score(validation_features,validation_data['safe_loans'])
validation_err_100 = 1 - ensemble_classifier_est_100.score(validation_features,validation_data['safe_loans'])
validation_err_200 = 1 - ensemble_classifier_est_200.score(validation_features,validation_data['safe_loans'])
validation_err_500 = 1 - ensemble_classifier_est_500.score(validation_features,validation_data['safe_loans'])
validation_errors = [validation_err_10, validation_err_50, validation_err_100, validation_err_200, validation_err_500]

plt.plot([10, 50, 100, 200, 500], training_errors, linewidth=4.0, label='Training error')
plt.plot([10, 50, 100, 200, 500], validation_errors, linewidth=4.0, label='Validation error')

make_figure(dim=(10,5), title='Error vs number of trees',
            xlabel='Number of trees',
            ylabel='Classification error',
            legend='best')