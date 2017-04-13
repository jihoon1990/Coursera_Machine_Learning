# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 15:17:59 2017

@author: Jihoon Kim

"""
import pandas as pd
from regression_functions import simple_linear_regression
from regression_functions import get_regression_predictions
from regression_functions import get_residual_sum_of_squares
from regression_functions import inverse_regression_predictions

# importing data type
dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

# Load Data from CSV files
sales = pd.read_csv("kc_house_data.csv", dtype = dtype_dict)
train_data = pd.read_csv("kc_house_train_data.csv", dtype = dtype_dict)
test_data = pd.read_csv("kc_house_test_data.csv", dtype = dtype_dict)

# save data for predicting price from sqft_living
sqft_intercept, sqft_slope = simple_linear_regression(train_data['sqft_living'], train_data['price'])

print("Intercept: " + str(sqft_intercept))
print("Slope    : " + str(sqft_slope))

# Predicts the price of a house with 2650 sqft
predicted_output = get_regression_predictions(2650,sqft_intercept,sqft_slope)
print("Predicted Price of a house with 2650 sqft: " + str(predicted_output))

# RSS for the simple linear regression using squarefeet to predict prices on TRAINING data
RSS_prices_on_sqft = get_residual_sum_of_squares(train_data['sqft_living'], train_data['price'], sqft_intercept, sqft_slope)
print("The RSS of predicting prices based on Square Feet is: %.5g " % RSS_prices_on_sqft)

# estimated square-feet for a house costing $800,000
estimate_sqft = inverse_regression_predictions(800000,sqft_intercept,sqft_slope)
print("Estimated Sqft for a house costing $800,000: %.5g" % estimate_sqft)


# New linear regression model with bedrooms
print()
# Estimate the slope and intercept for predicting 'price' based on 'bedrooms'
bedrooms_intercept, bedrooms_slope = simple_linear_regression(train_data['bedrooms'], train_data['price'])
print("Intercept: " + str(bedrooms_intercept))
print("Slope: " + str(bedrooms_slope))

# Test Linear Regression Algorithm
# Compute RSS when using bedrooms on TEST data:
rss_prices_on_bedrooms = get_residual_sum_of_squares(test_data['bedrooms'], test_data['price'], bedrooms_intercept, bedrooms_slope)
print('The RSS of predicting Prices based on Square Feet is : %.5g' % rss_prices_on_bedrooms)
# Compute RSS when using squarefeet on TEST data:
rss_prices_on_sqft = get_residual_sum_of_squares(test_data['sqft_living'], test_data['price'], sqft_intercept, sqft_slope)
print('The RSS of predicting Prices based on Square Feet is : %.5g' % rss_prices_on_sqft)