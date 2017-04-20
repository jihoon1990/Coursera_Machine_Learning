# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 10:56:33 2017

@author: Jihoon_Kim
"""
import string
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from linear_classifier_func import calculate_probability
#------Data Cleansing------
# Load Data
products = pd.read_csv('amazon_baby.csv')
# Fill N/A
products = products.fillna({'review':''})  # fill in N/A's in the review column
# Reomove Punctuation for Text Cleaning
products['review'] = products['review'].astype('str')
products['review_clean'] = products['review'].apply(lambda x: ''.join([i for i in x if i not in string.punctuation]))
# Add Sentiment Column
products['sentiment'] = products['rating'].apply(lambda rating: 1 if rating >= 4 else -1)

# Extract Train Data
train_index = pd.read_json('train-idx.json')
train_idx_list = train_index[0].values.tolist()
train_data = products.ix[train_idx_list]
# Extract Test Data
test_index = pd.read_json('test-idx.json')
test_idx_list = test_index[0].values.tolist()
test_data = products.ix[test_idx_list]

# Remove raing of 3 since it is neutral
products = products[products['rating'] != 3]
train_data = train_data[train_data['rating'] != 3]
test_data = test_data[test_data['rating'] != 3]

print("[DATA CLEANSING: Done]")

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
sentiment_model = LogisticRegression()
sentiment_model.fit(train_matrix,train_data['sentiment'])
print("[SENTIMENT_MODEL: DONE]\n")

# Number of positive coefficients
print("Number of positive coefficients: ", sum(sentiment_model.coef_[0] >= 0))

# Making predictions with logistic regression
sample_test_data = test_data[10:13]
print("Sample Test Data: ", sample_test_data)

# Let's explore some test data
print(sample_test_data)

# Predict score of sample_test_data
print()
sample_test_matrix = vectorizer.transform(sample_test_data['review_clean'])
scores = sentiment_model.decision_function(sample_test_matrix)
print("Score of sample test data: ", scores)

# Predicting Sentiment
print("Predicted sentiment of sample data: ", sentiment_model.predict(sample_test_matrix))

# Probability Calc
print("Probability Prediction: ", calculate_probability(scores))

# Find the most positive (and negative) review
# probability predictions on test_data using the sentiment_model
test_scores = sentiment_model.decision_function(test_matrix)
test_data['prob'] = calculate_probability(test_scores)
# Let's explore test_data
print(test_data.head(5))

# 20 Most Positive Reviews:
print("20 Most Positive Reviews: \n", test_data[['name','prob']].sort_values(by='prob',ascending=False).head(20))
# 20 Most Negative Reviews:
print("20 Most Negative Reviews: \n", test_data[['name','prob']].sort_values(by='prob',ascending=True).head(20))

# Compute accuracy of the classifier
print()
print("ACCURACY: ", (test_data['sentiment'] == sentiment_model.predict(test_matrix)).sum()/len(test_data))

# Learn another classifier with fewer words
significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves', 
      'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed', 
      'work', 'product', 'money', 'would', 'return']

vectorizer_word_subset = CountVectorizer(vocabulary=significant_words) # limit to 20 words
train_matrix_word_subset = vectorizer_word_subset.fit_transform(train_data['review_clean'])
test_matrix_word_subset = vectorizer_word_subset.transform(test_data['review_clean'])

# Train a logistic regression model on a subset of data
simple_model = LogisticRegression()
simple_model.fit(train_matrix_word_subset,train_data['sentiment'])
# Number of positive coefficients
print("Number of positive coefficients: ", sum(simple_model.coef_[0] >= 0))

# Explore Coefficients
print(pd.DataFrame(list(zip(significant_words,simple_model.coef_[0])), columns = ['features', 'estimated coefficients']))

# Comparing models
print("ACCURACY (Sentiment Model, TRAIN DATA): ", (train_data['sentiment'] == sentiment_model.predict(train_matrix)).sum()/len(train_data))
print("ACCURACY (Simple Model, TRAIN DATA): ", (train_data['sentiment'] == simple_model.predict(train_matrix_word_subset)).sum()/len(train_data))
print()
print("ACCURACY (Sentiment Model, TEST DATA): ", (test_data['sentiment'] == sentiment_model.predict(test_matrix)).sum()/len(test_data))
print("ACCURACY (Simple Model, TEST DATA): ", (test_data['sentiment'] == simple_model.predict(test_matrix_word_subset)).sum()/len(test_data))

# Baseline: Majority class prediction
# Majority Class:
print()
print("MAJORITY CLASS MODEL")
num_positive  = (train_data['sentiment'] == +1).sum()
num_negative = (train_data['sentiment'] == -1).sum()
print("# Positive: ", num_positive)
print("# Negative: ", num_negative)
print("ACCURACY (Majority Class Model, TEST DATA): ", (test_data['sentiment']==1).sum()/len(test_data))