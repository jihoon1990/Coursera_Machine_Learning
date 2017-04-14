# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 15:37:11 2017

@author: Jihoon_Kim
"""

# Import Module
import pandas as pd
import numpy as np
from sklearn import linear_model
from math import log

# Import Data Type
dtype_dict = {'bathrooms':float, 'waterfront':int, 
              'sqft_above':int, 'sqft_living15':float, 
              'grade':int, 'yr_renovated':int, 'price':float, 
              'bedrooms':float, 'zipcode':str, 'long':float, 
              'sqft_lot15':float, 'sqft_living':float, 
              'floors':str, 'condition':int, 'lat':float, 
              'date':str, 'sqft_basement':int, 'yr_built':int, 
              'id':str, 'sqft_lot':int, 'view':int}

# Load Data from CSV files
sales = pd.read_csv("kc_house_data.csv", dtype = dtype_dict)
train_data = pd.read_csv("kc_house_train_data.csv", dtype = dtype_dict)
test_data = pd.read_csv("kc_house_test_data.csv", dtype = dtype_dict)

# 4 new features
# * bedrooms_squared = bedrooms * bedrooms
# * bed_bath_rooms = bedrooms * bathrooms
# * log_sqft_living = log(sqft_living)
# * lat_plus_long = lat + long

train_data['bedrooms_squared'] = train_data['bedrooms'].apply(lambda x: x**2)
test_data['bedrooms_squared'] = test_data['bedrooms'].apply(lambda x: x**2)

train_data['bed_bath_rooms'] = train_data['bedrooms'] * train_data['bathrooms']
test_data['bed_bath_rooms'] = test_data['bedrooms'] * test_data['bathrooms']

train_data['log_sqft_living'] = train_data['sqft_living'].apply(lambda x: log(x))
test_data['log_sqft_living'] = test_data['sqft_living'].apply(lambda x: log(x))

train_data['lat_plus_long'] = train_data['lat'] + train_data['long']
test_data['lat_plus_long'] = test_data['lat'] + test_data['long']

# Explore new features
print(train_data[['bedrooms','bathrooms','lat','long','bedrooms_squared',
            'bed_bath_rooms','log_sqft_living','lat_plus_long']].head())

# Average value of new features on TEST Data
print()
print('Average of \'bedrooms_squared\': ' + str(test_data['bedrooms_squared'].mean()))
print('Average of \'bed_bath_rooms\': ' + str(test_data['bed_bath_rooms'].mean()))
print('Average of \'log_sqft_living\': ' + str(test_data['log_sqft_living'].mean()))
print('Average of \'lat_plus_long\': ' + str(test_data['lat_plus_long'].mean()))

# Learning Multiple Linear Regression Model
# Model 1: squarefeet, # bedrooms, # bathrooms, latitude & longitude
# Model 2: Model 1 features + bed_bath_rooms
# Model 3: Model 2 features + bedrooms_squared, log_sqft_living, lat_plus_long

print()
print("MODEL 1")
model_1 = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
model_1_x_train = train_data[['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long']]
model_1_y_train = train_data['price']
model_1.fit(model_1_x_train,model_1_y_train)
print("Intercepts: ", model_1.intercept_)
coeffs = pd.DataFrame(list(zip(model_1_x_train.columns,model_1.coef_)), columns = ['features', 'estimated coefficients'])
print(coeffs)

print()
print("MODEL 2")
model_2 = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
model_2_x_train = train_data[['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long', 'bed_bath_rooms']]
model_2_y_train = train_data['price']
model_2.fit(model_2_x_train,model_2_y_train)
print("Intercepts: ", model_2.intercept_)
coeffs = pd.DataFrame(list(zip(model_2_x_train.columns,model_2.coef_)), columns = ['features', 'estimated coefficients'])
print(coeffs)

print()
print("MODEL 3")
model_3 = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
model_3_x_train = train_data[['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long', 'bed_bath_rooms', 'bedrooms_squared', 'log_sqft_living', 'lat_plus_long']]
model_3_y_train = train_data['price']
model_3.fit(model_3_x_train,model_3_y_train)
print("Intercepts: ", model_3.intercept_)
coeffs = pd.DataFrame(list(zip(model_3_x_train.columns,model_3.coef_)), columns = ['features', 'estimated coefficients'])
print(coeffs)

## Assign Test Data Set
model_1_x_test = test_data[['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long']]
model_1_y_test = test_data['price']
model_2_x_test = test_data[['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long', 'bed_bath_rooms']]
model_2_y_test = test_data['price']
model_3_x_test = test_data[['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long', 'bed_bath_rooms', 'bedrooms_squared', 'log_sqft_living', 'lat_plus_long']]
model_3_y_test = test_data['price']

## Evaluation  
print()
model_1_rss_train = np.sum((model_1_y_train - model_1.predict(model_1_x_train))**2)
print("RSS of Model 1 (train data): %.4g" % model_1_rss_train)
model_2_rss_train = np.sum((model_2_y_train - model_2.predict(model_2_x_train))**2)
print("RSS of Model 2 (train data): %.4g" % model_2_rss_train)
model_3_rss_train = np.sum((model_3_y_train - model_3.predict(model_3_x_train))**2)
print("RSS of Model 3 (train data): %.4g" % model_3_rss_train)

print()
model_1_rss_test = np.sum((model_1_y_test - model_1.predict(model_1_x_test))**2)
print("RSS of Model 1 (test data): %.4g" % model_1_rss_test)
model_2_rss_test = np.sum((model_2_y_test - model_2.predict(model_2_x_test))**2)
print("RSS of Model 2 (test data): %.4g" % model_2_rss_test)
model_3_rss_test = np.sum((model_3_y_test - model_3.predict(model_3_x_test))**2)
print("RSS of Model 3 (test data): %.4g" % model_3_rss_test)