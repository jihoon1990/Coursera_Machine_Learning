# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 19:17:31 2017

@author: Jihoon_Kim
"""

# Logistic Regression via Stochastic Gradient Ascent

# Import Modules
import string
import pandas as pd
import numpy as np

from logistic_regression_via_stochastic_gradient_descent_func import predict_probability
from logistic_regression_via_stochastic_gradient_descent_func import feature_derivative
from logistic_regression_via_stochastic_gradient_descent_func import compute_avg_log_likelihood
from logistic_regression_via_stochastic_gradient_descent_func import logistic_regression_SG
from logistic_regression_via_stochastic_gradient_descent_func import make_plot 
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
    

# Convert data frame to multi-dimensional array
products['intercept'] = 1
# Loat train-test data
train_idx = pd.read_json("train-idx.json")
validation_idx = pd.read_json("validation-idx.json")
train_data = products.iloc[train_idx[0].values]
validation_data = products.iloc[validation_idx[0].values]

feature_set = ['intercept'] + important_words[0].tolist()
# Coverrt to ndarray
feature_matrix_train = train_data[feature_set].as_matrix()
sentiment_train = train_data['sentiment'].as_matrix()
feature_matrix_valid = validation_data[feature_set].as_matrix()
sentiment_valid = validation_data['sentiment'].as_matrix()
# Shape of feature matrix
print("Size of feature matrix (TRAIN): ", feature_matrix_train.shape)
print("Size of feature matrix (VALIDATION): ", feature_matrix_valid.shape)


# Modifying the derivative for stochastic gradient ascent
# Computing the gradient for a single data point
j = 1                        # Feature number
i = 10                       # Data point number
coefficients = np.zeros(194) # A point w at which we are computing the gradient.

predictions = predict_probability(feature_matrix_train[i:i+1,:], coefficients)
indicator = (sentiment_train[i:i+1]==+1)

errors = indicator - predictions
gradient_single_data_point = feature_derivative(errors, feature_matrix_train[i:i+1,j])
print("Gradient single data point: %s" % gradient_single_data_point)
print("           --> Should print 0.0")

# Modifying the derivative for using a batch of data points
# Computing the gradient for a "mini-batch" of data points

j = 1                        # Feature number
i = 10                       # Data point start
B = 10                       # Mini-batch size
coefficients = np.zeros(194) # A point w at which we are computing the gradient.

predictions = predict_probability(feature_matrix_train[i:i+B,:], coefficients)
indicator = (sentiment_train[i:i+B]==+1)

errors = indicator - predictions        
gradient_mini_batch = feature_derivative(errors, feature_matrix_train[i:i+B,j])
print("Gradient mini-batch data points: %s" % gradient_mini_batch)
print("                --> Should print 1.0")


# Implementing stochastic gradient ascent
sample_feature_matrix = np.array([[1.,2.,-1.], [1.,0.,1.]])
sample_sentiment = np.array([+1, -1])

coefficients, log_likelihood = logistic_regression_SG(sample_feature_matrix, sample_sentiment, np.zeros(3),
                                                  step_size=1., batch_size=2, max_iter=2)
print('-------------------------------------------------------------------------------------')
print('Coefficients learned                 :', coefficients)
print('Average log likelihood per-iteration :', log_likelihood)
if np.allclose(coefficients, np.array([-0.09755757,  0.68242552, -0.7799831]), atol=1e-3)\
  and np.allclose(log_likelihood, np.array([-0.33774513108142956, -0.2345530939410341])):
    # pass if elements match within 1e-3
    print('-------------------------------------------------------------------------------------')
    print('Test passed!')
else:
    print('-------------------------------------------------------------------------------------')
    print('Test failed')
    
# Running gradient ascent using the stochastic gradient ascent implementation
coefficients, log_likelihood = logistic_regression_SG(feature_matrix_train, sentiment_train,
                                        initial_coefficients=np.zeros(194),
                                        step_size=5e-1, batch_size=1, max_iter=10)

coefficients_batch, log_likelihood_batch = logistic_regression_SG(feature_matrix_train, sentiment_train,
                                        initial_coefficients=np.zeros(194),
                                        step_size=5e-1, 
                                        batch_size = len(feature_matrix_train), 
                                        max_iter=200)

# Log likelihood plots for stochastic gradient ascent
step_size = 1e-1
batch_size = 100
num_passes = 10
num_iterations = num_passes * int(len(feature_matrix_train)/batch_size)

coefficients_sgd, log_likelihood_sgd = logistic_regression_SG(feature_matrix_train, sentiment_train,
                                       initial_coefficients=np.zeros(194),
                                       step_size=1e-1, batch_size=100, max_iter=num_iterations)

make_plot(log_likelihood_sgd, len_data=len(feature_matrix_train), batch_size=100,
          label='stochastic gradient, step_size=1e-1')

# Smoothing the stochastic gradient ascent curve
make_plot(log_likelihood_sgd, len_data=len(feature_matrix_train), batch_size=100,
          smoothing_window=30, label='stochastic gradient, step_size=1e-1')

# Stochastic gradient ascent vs batch gradient ascent
step_size = 1e-1
batch_size = 100
num_passes = 200
num_iterations = num_passes * int(len(feature_matrix_train)/batch_size)

## YOUR CODE HERE
coefficients_sgd, log_likelihood_sgd = logistic_regression_SG(feature_matrix_train, sentiment_train,
                                       initial_coefficients=np.zeros(194),
                                       step_size=step_size, batch_size=batch_size, max_iter=num_iterations)

make_plot(log_likelihood_sgd, len_data=len(feature_matrix_train), batch_size=100,
          smoothing_window=30, label='stochastic, step_size=1e-1')
make_plot(log_likelihood_batch, len_data=len(feature_matrix_train), batch_size=len(feature_matrix_train),
          smoothing_window=1, label='batch, step_size=5e-1')

# Explore the effects of step sizes on stochastic gradient ascent
batch_size = 100
num_passes = 10
num_iterations = num_passes * int(len(feature_matrix_train)/batch_size)

coefficients_sgd = {}
log_likelihood_sgd = {}
for step_size in np.logspace(-4, 2, num=7):
    coefficients_sgd[step_size], log_likelihood_sgd[step_size] = logistic_regression_SG(feature_matrix_train, sentiment_train,
                                       initial_coefficients=np.zeros(194),
                                       step_size=step_size, batch_size=batch_size, max_iter=num_iterations)
    
# Plotting the log likelihood as a function of passes for each step size

for step_size in np.logspace(-4, 2, num=7):
    make_plot(log_likelihood_sgd[step_size], len_data=len(train_data), batch_size=100,
              smoothing_window=30, label='step_size=%.1e'%step_size)
    


for step_size in np.logspace(-4, 2, num=7)[0:6]:
    make_plot(log_likelihood_sgd[step_size], len_data=len(train_data), batch_size=100,
              smoothing_window=30, label='step_size=%.1e'%step_size)