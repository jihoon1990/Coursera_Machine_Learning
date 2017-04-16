# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 12:20:11 2017

@author: Jihoon_Kim
"""

# Ridge Regression Functions
import pandas as pd
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
    model = linear_model.Ridge(alpha = alpha)
    model.fit(x,y)
    print("Intercepts: ", model.intercept_)
    coeffs = pd.DataFrame(list(zip(x.columns,model.coef_)), columns = ['features', 'estimated coefficients'])
    print(coeffs)
    return model
    
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