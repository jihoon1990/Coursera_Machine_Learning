# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 17:23:02 2017

@author: Jihoon_Kim
"""

# Logistic Regeression with L2 Regularization

# Import Modules
import string
import numpy as np
import pandas as pd

from logistic_classifier_func import logistic_regression_with_L2
from logistic_classifier_func import make_coefficient_plot
from logistic_classifier_func import get_classification_accuracy

# Load Data
products = pd.read_csv('amazon_baby_subset.csv')
# Fill N/A
products = products.fillna({'review':''})  # fill in N/A's in the review column
# Reomove Punctuation for Text Cleaning
products['review'] = products['review'].astype('str')
products['review_clean'] = products['review'].apply(lambda x: ''.join([i for i in x if i not in string.punctuation]))
# Add Sentiment Column
products['sentiment'] = products['rating'].apply(lambda rating: 1 if rating >= 4 else -1)

# Import Important words
important_words = pd.read_json('important_words.json')

# Counting Important words in `review_clean` column
for word in important_words[0].values.tolist():
    products[word] = products['review_clean'].apply(lambda s : s.split().count(word))

# Extract Train Data
train_index = pd.read_json('train-idx.json')
train_idx_list = train_index[0].values.tolist()
train_data = products.ix[train_idx_list]
# Extract Test Data
valid_index = pd.read_json('validation-idx.json')
valid_idx_list = valid_index[0].values.tolist()
valid_data = products.ix[valid_idx_list]

# Train-Validation Split
print("Train Data: \n", train_data.head(5))
print("Validation Data \n", valid_data.head(5))
print("Number of Train Data: ", len(train_data))
print("Number of Validation Data: ", len(valid_data))

# Convert to Numpy Array
feature_set = ['intercept'] + important_words[0].tolist()
## Train Data
train_data['intercept'] = 1
feature_matrix_train = train_data[feature_set].as_matrix()
sentiment_train = train_data['sentiment'].as_matrix()
# Shape of feature matrix
print("Size of feature matrix: ", feature_matrix_train.shape)

## Validation Data
valid_data['intercept'] = 1
feature_matrix_valid = valid_data[feature_set].as_matrix()
sentiment_valid = valid_data['sentiment'].as_matrix()
# Shape of feature matrix
print("Size of feature matrix: ", feature_matrix_valid.shape)

# Explore effects of L2 regularization
# run with L2 = 0
coefficients_0_penalty = logistic_regression_with_L2(feature_matrix_train, sentiment_train,
                                                     initial_coefficients=np.zeros(194),
                                                     step_size=5e-6, l2_penalty=0, max_iter=501)
# run with L2 = 4
coefficients_4_penalty = logistic_regression_with_L2(feature_matrix_train, sentiment_train,
                                                      initial_coefficients=np.zeros(194),
                                                      step_size=5e-6, l2_penalty=4, max_iter=501)

# run with L2 = 10
coefficients_10_penalty = logistic_regression_with_L2(feature_matrix_train, sentiment_train,
                                                      initial_coefficients=np.zeros(194),
                                                      step_size=5e-6, l2_penalty=10, max_iter=501)
# run with L2 = 1e2
coefficients_1e2_penalty = logistic_regression_with_L2(feature_matrix_train, sentiment_train,
                                                       initial_coefficients=np.zeros(194),
                                                       step_size=5e-6, l2_penalty=1e2, max_iter=501)
# run with L2 = 1e3
coefficients_1e3_penalty = logistic_regression_with_L2(feature_matrix_train, sentiment_train,
                                                       initial_coefficients=np.zeros(194),
                                                       step_size=5e-6, l2_penalty=1e3, max_iter=501)
# run with L2 = 1e5
coefficients_1e5_penalty = logistic_regression_with_L2(feature_matrix_train, sentiment_train,
                                                       initial_coefficients=np.zeros(194),
                                                       step_size=5e-6, l2_penalty=1e5, max_iter=501)

# Compare coefficients
table = pd.DataFrame(feature_set)
table = table.rename(columns={0:'words'})

def add_coefficients_to_table(coefficients, column_name):
    table[column_name] = coefficients
    return table

add_coefficients_to_table(coefficients_0_penalty, 'coefficients [L2=0]')
add_coefficients_to_table(coefficients_4_penalty, 'coefficients [L2=4]')
add_coefficients_to_table(coefficients_10_penalty, 'coefficients [L2=10]')
add_coefficients_to_table(coefficients_1e2_penalty, 'coefficients [L2=1e2]')
add_coefficients_to_table(coefficients_1e3_penalty, 'coefficients [L2=1e3]')
add_coefficients_to_table(coefficients_1e5_penalty, 'coefficients [L2=1e5]')

# Using the coefficients trained with L2 penalty 0, find the 5 most positive words (with largest positive coefficients). Save them to positive_words. Similarly, find the 5 most negative words (with largest negative coefficients) and save them to negative_words.
print()
positive_words = table.sort_values(by='coefficients [L2=0]', ascending = False)[0:5]['words']
print("Five most positive words (L2=0):\n", positive_words)

negative_words = table.sort_values(by='coefficients [L2=0]', ascending = True)[0:5]['words']
print("Five most negative words (L2=0):\n", negative_words)

# Compare coefficients
make_coefficient_plot(table, positive_words, negative_words, l2_penalty_list=[0, 4, 10, 1e2, 1e3, 1e5])

# Measuring accuracy
train_accuracy = {}
train_accuracy[0]   = get_classification_accuracy(feature_matrix_train, sentiment_train, coefficients_0_penalty)
train_accuracy[4]   = get_classification_accuracy(feature_matrix_train, sentiment_train, coefficients_4_penalty)
train_accuracy[10]  = get_classification_accuracy(feature_matrix_train, sentiment_train, coefficients_10_penalty)
train_accuracy[1e2] = get_classification_accuracy(feature_matrix_train, sentiment_train, coefficients_1e2_penalty)
train_accuracy[1e3] = get_classification_accuracy(feature_matrix_train, sentiment_train, coefficients_1e3_penalty)
train_accuracy[1e5] = get_classification_accuracy(feature_matrix_train, sentiment_train, coefficients_1e5_penalty)

validation_accuracy = {}
validation_accuracy[0]   = get_classification_accuracy(feature_matrix_valid, sentiment_valid, coefficients_0_penalty)
validation_accuracy[4]   = get_classification_accuracy(feature_matrix_valid, sentiment_valid, coefficients_4_penalty)
validation_accuracy[10]  = get_classification_accuracy(feature_matrix_valid, sentiment_valid, coefficients_10_penalty)
validation_accuracy[1e2] = get_classification_accuracy(feature_matrix_valid, sentiment_valid, coefficients_1e2_penalty)
validation_accuracy[1e3] = get_classification_accuracy(feature_matrix_valid, sentiment_valid, coefficients_1e3_penalty)
validation_accuracy[1e5] = get_classification_accuracy(feature_matrix_valid, sentiment_valid, coefficients_1e5_penalty)

# Build a simple report
for key in sorted(validation_accuracy.keys()):
    print("L2 penalty = %g" % key)
    print("train accuracy = %s, validation_accuracy = %s" % (train_accuracy[key], validation_accuracy[key]))
    print("---------------------------------------------------------------------")