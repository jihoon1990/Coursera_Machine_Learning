# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 12:20:11 2017

@author: Jihoon_Kim
"""

# Ridge Regression Functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

def polynomial_dataframe(feature, degree): # feature is pandas.Series type
    # assume that degree >= 1
    # initialize the dataframe:
    poly_dataframe = pd.DataFrame()
    # and set poly_dataframe['power_1'] equal to the passed feature
    poly_dataframe['power_1'] = feature
    # first check if degree > 1
    if degree > 1:
        # then loop over the remaining degrees:
        for power in range(2, degree+1):
            # first we'll give the column a name:
            name = 'power_' + str(power)
            # assign poly_dataframe[name] to be feature^power; use apply(*)
            poly_dataframe[name] = feature.apply(lambda x: x ** power)
    return poly_dataframe

def ridge_model(alpha,x,y):
    model = linear_model.Ridge(alpha = alpha, normalize = True)
    model.fit(x,y)
    return model

    print("Intercepts: ", model.intercept_)
    coeffs = pd.DataFrame(list(zip(x.columns,model.coef_)), columns = ['features', 'estimated coefficients'])
    print(coeffs)
    
def plot_fitted_line(n,set_data,model):
    plt.figure(n)
    plt.plot(set_data['power_1'], set_data['price'],'.',
             set_data['power_1'], model.predict(set_data.drop('price',1)),'-')
    plt.show()
    
def k_fold_cross_validation(k, l2_penalty, data):    
    rss_sum = 0
    n = len(data)
    for i in range(k):
        start = int((n*i)/k)
        end = int((n*(i+1))/k-1)
        validation_set = data[start:end+1]
        training_set = data[0:start].append(data[end+1:n])    
        model = linear_model.Ridge(alpha = l2_penalty)
        model.fit(training_set.drop('price',1),training_set['price'])
        predictions = model.predict(validation_set.drop('price',1))
        residuals = validation_set['price'] - predictions
        rss = sum(residuals * residuals)
        rss_sum += rss
    validation_error = rss_sum / k # average = sum / size or you can use np.mean(list_of_validation_error)
    return validation_error

def predict_output(feature_matrix, weights):
    # assume feature_matrix is a numpy matrix containing the features as columns and weights is a corresponding numpy array
    # create the predictions vector by using np.dot()
    predictions = np.dot(feature_matrix, weights)
    return predictions

def feature_derivative_ridge(errors, feature, weight, l2_penalty, feature_is_constant):
    # if feature_is_constant is True, derivative is twice the dot product of errors and feature
    if feature_is_constant == True:
        derivative = 2*np.dot(errors,feature)
    # otherwise, derivative is twice the dot product plus 2*l2_penalty*weight
    else:
        derivative = 2*np.dot(errors,feature) + 2*(l2_penalty*weight)
    return derivative

def ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations = 100):
    weights = np.array(initial_weights)
    
    # while not reached maximum number of iterations:
    while max_iterations > 0:
        # compute the predictions based on feature_matrix and wieights using predict_output() function
        predictions = predict_output(feature_matrix,weights)
        # compute the errors as predictions - output
        errors = predictions - output
        for i in range(len(weights)):
            # Recall that feature_matrix[:,i] is the feature column associated with weights[i]
            # compute the derivative for weight[i].
            #(Remember: when i=0, you are computing the derivative of the constant!)
            if i == 0:
                feature_is_constant = True
            else:
                feature_is_constant = False
            derivative = feature_derivative_ridge(errors, feature_matrix[:,i], weights[i], l2_penalty, feature_is_constant)
            weights[i] = weights[i] - (step_size * derivative)
        max_iterations -= 1
    return weights            