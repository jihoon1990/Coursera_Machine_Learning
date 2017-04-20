# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 13:29:39 2017

@author: Jihoon_Kim
"""

# Logistic Regeression (Implementation)

# Import Modules
import string
import numpy as np
import pandas as pd
from logistic_classifier_func import logistic_regression

# Load Data
products = pd.read_csv('amazon_baby_subset.csv')
# Fill N/A
products = products.fillna({'review':''})  # fill in N/A's in the review column
# Reomove Punctuation for Text Cleaning
products['review'] = products['review'].astype('str')
products['review_clean'] = products['review'].apply(lambda x: ''.join([i for i in x if i not in string.punctuation]))
# Add Sentiment Column
products['sentiment'] = products['rating'].apply(lambda rating: 1 if rating >= 4 else -1)

# Explore data: Name
print(products['name'].head(10))

# Counting Positive and Negative Reviews
print("\nCounting Positive & Negative Reviews")
num_pos = sum(products['sentiment'] == 1)
num_neg = sum(products['sentiment'] == -1)
print("Number of Positive Reviews: ", num_pos)
print("Number of Negative Reviews: ", num_neg)

# Import Important words
important_words = pd.read_json('important_words.json')
print("Important Words: ", important_words)

# Counting Important words in `review_clean` column
for word in important_words[0].values.tolist():
    products[word] = products['review_clean'].apply(lambda s : s.split().count(word))

# Now column ['perfect'] contains the count of the word 'perfect' occured in each reviews.
# Number of reviews contain 'perfect'
products['contains_perfect'] = products['perfect'].apply(lambda x: 1 if x >= 1 else 0)
print("Number of reviews contain 'perfect': ", products['contains_perfect'].sum())

# Convert data frame to multi-dimensional array
products['intercept'] = 1
feature_set = ['intercept'] + important_words[0].tolist()
# Coverrt to ndarray
feature_matrix = products[feature_set].as_matrix()
sentiment = products['sentiment'].as_matrix()

# Shape of feature matrix
print("Size of feature matrix: ", feature_matrix.shape)

# Taking Gradient Steps
coefficients = logistic_regression(feature_matrix, sentiment, initial_coefficients=np.zeros(194),
                                   step_size=1e-7, max_iter=301)

# Predicting sentiments
# Now, let's apply the coefficients to predict sentiment
scores = np.dot(feature_matrix, coefficients)
class_predictions = pd.DataFrame(scores)[0].apply(lambda x: 1 if x > 0 else -1)
# Predicted Positive Sentiment
print("Predicted Positive Sentiment: ", (class_predictions.values > 0).sum())

# Measuring accuracy
num_mistakes = (class_predictions != sentiment).sum() # YOUR CODE HERE
num_correct = len(sentiment) - num_mistakes
accuracy = num_correct/len(sentiment) # YOUR CODE HERE
print("-----------------------------------------------------")
print('# Reviews   correctly classified =', num_correct)
print('# Reviews incorrectly classified =', num_mistakes)
print('# Reviews total                  =', len(products))
print("-----------------------------------------------------")
print('Accuracy = %.2f' % accuracy)

# Which words contribute most to positive & negative sentiments
coefficients = list(coefficients[1:]) # exclude intercept
word_coefficient_tuples = [(word, coefficient) for word, coefficient in zip(important_words[0].values, coefficients)]
word_coefficient_tuples = sorted(word_coefficient_tuples, key=lambda x:x[1], reverse=True)
# Most Positive Words
print("Most Positive Words: \n", word_coefficient_tuples[:10])
# Most Negative Words
print("Most Negative Words: \n", word_coefficient_tuples[-10:])