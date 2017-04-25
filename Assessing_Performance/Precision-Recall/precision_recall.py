# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 16:30:23 2017

@author: Jihoon_Kim
"""
# Precision-Recall

# Import Module
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from precision_recall_func import apply_threshold
from precision_recall_func import plot_pr_curve
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Load Data
products = pd.read_csv('amazon_baby.csv')

# Reomove Punctuation for Text Cleaning
products['review'] = products['review'].astype('str')
products['review_clean'] = products['review'].apply(lambda x: ''.join([i for i in x if i not in string.punctuation]))
# Add Sentiment Column
products['sentiment'] = products['rating'].apply(lambda rating: 1 if rating > 3 else -1)

# Quick peek
print(products.head(5))

# Loat train-test data
train_idx = pd.read_json("train-idx.json")
test_idx = pd.read_json("test-idx.json")
train_data = products.iloc[train_idx[0].values]
test_data = products.iloc[test_idx[0].values]

# Remove raing of 3 since it is neutral
products = products[products['rating'] != 3]
train_data = train_data[train_data['rating'] != 3]
test_data = test_data[test_data['rating'] != 3]

#------Generate Bag of Words Vector------
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
# Use this token pattern to keep single-letter words
# First, learn vocabulary from the training data and assign columns to words
# Then convert the training data into a sparse matrix
train_matrix = vectorizer.fit_transform(train_data['review_clean'])
# Second, convert the test data into a sparse matrix, using the same word-column mapping
test_matrix = vectorizer.transform(test_data['review_clean'])
print("[GENERATING WORD COUNTING VECTOR: DONE]")

#------Train a sentiment classifier with logistic regression------
# Sentiment Model
model = LogisticRegression()
model.fit(train_matrix,train_data['sentiment'])
print("[SENTIMENT_MODEL: DONE]\n")

accuracy = accuracy_score(y_true=test_data['sentiment'], y_pred=model.predict(test_matrix))
print("Test Accuracy: %s" % accuracy)

# Baseline: Majority class prediction
baseline = len(test_data[test_data['sentiment'] == 1])/len(test_data)
print("Baseline accuracy (majority class classifier): %s" % baseline)

# Confusion Matrix
cmat = confusion_matrix(y_true=test_data['sentiment'],
                        y_pred=model.predict(test_matrix),
                        labels=model.classes_)    # use the same order of class as the LR model.
print(' target_label | predicted_label | count ')
print('--------------+-----------------+-------')
# Print out the confusion matrix.
# NOTE: Your tool may arrange entries in a different order. Consult appropriate manuals.
for i, target_label in enumerate(model.classes_):
    for j, predicted_label in enumerate(model.classes_):
        print('{0:^13} | {1:^15} | {2:5d}'.format(target_label, predicted_label, cmat[i,j]))
        
# Precision and Recall
precision = precision_score(y_true=test_data['sentiment'], 
                            y_pred=model.predict(test_matrix))
print("Precision on test data: %s" % precision)

recall = recall_score(y_true=test_data['sentiment'],
                      y_pred=model.predict(test_matrix))
print("Recall on test data: %s" % recall)

# Precision-recall tradeoff
# Explore apply_threshold function
probabilities = model.predict_proba(test_matrix)[:,1]
predictions_with_default_threshold = apply_threshold(probabilities, 0.5)
predictions_with_high_threshold = apply_threshold(probabilities, 0.9)
print("Number of positive predicted reviews (threshold = 0.5): %s" % (predictions_with_default_threshold == 1).sum())
print("Number of positive predicted reviews (threshold = 0.9): %s" % (predictions_with_high_threshold == 1).sum())

# Exploring the associated precision and recall as the threshold varies
# Threshold = 0.5
precision_with_default_threshold = precision_score(y_true=test_data['sentiment'], 
                            y_pred=predictions_with_default_threshold)
print("Precision (threshold = 0.5): %s" % precision_with_default_threshold)

recall_with_default_threshold = recall_score(y_true=test_data['sentiment'],
                      y_pred=predictions_with_default_threshold)
print("Recall (threshold = 0.5)   : %s" % recall_with_default_threshold)

# Threshold = 0.9
precision_with_high_threshold = precision_score(y_true=test_data['sentiment'], 
                            y_pred=predictions_with_high_threshold)
print("Precision (threshold = 0.9): %s" % precision_with_high_threshold)

recall_with_high_threshold = recall_score(y_true=test_data['sentiment'],
                      y_pred=predictions_with_high_threshold)
print("Recall (threshold = 0.9)   : %s" % recall_with_high_threshold)

# Precision-recall curve
threshold_values = np.linspace(0.5, 1, num=100)
print(threshold_values)

precision_all = []
recall_all = []
probabilities = model.predict_proba(test_matrix)[:,1]
for threshold in threshold_values:
    predictions = apply_threshold(probabilities, threshold)
    
    precision = precision_score(y_true=test_data['sentiment'], y_pred = predictions)
    recall = recall_score(y_true=test_data['sentiment'], y_pred = predictions)
    
    precision_all.append(precision)
    recall_all.append(recall)
    
plt.figure()
plot_pr_curve(precision_all, recall_all, 'Precision recall curve (all)')

# Evaluating specific search terms
baby_reviews = test_data[test_data['name'].str.lower().str.contains('baby',na=False)]
baby_matrix = vectorizer.transform(baby_reviews['review_clean'])
# Precision-recall curve
threshold_values = np.linspace(0.5, 1, num=100)
precision_all = []
recall_all = []
probabilities = model.predict_proba(baby_matrix)[:,1]

for threshold in threshold_values:
    predictions = apply_threshold(probabilities, threshold)
    
    precision = precision_score(y_true=baby_reviews['sentiment'], y_pred = predictions)
    recall = recall_score(y_true=baby_reviews['sentiment'], y_pred = predictions)
    
    precision_all.append(precision)
    recall_all.append(recall)

plt.figure()
plot_pr_curve(precision_all, recall_all, 'Precision recall curve (Baby)')