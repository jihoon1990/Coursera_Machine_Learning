# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 11:25:33 2017

@author: Jihoon_Kim
"""

# Import Module & Load Data
import pandas as pd
import numpy as np
from math import sqrt
from sklearn import linear_model  # using scikit-learn

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

sales = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)
testing = pd.read_csv('kc_house_test_data.csv', dtype=dtype_dict)
training = pd.read_csv('kc_house_train_data.csv', dtype=dtype_dict)
validation = pd.read_csv('kc_house_valid_data.csv', dtype=dtype_dict)

# Creating New Features
sales['sqft_living_sqrt'] = sales['sqft_living'].apply(sqrt)
sales['sqft_lot_sqrt'] = sales['sqft_lot'].apply(sqrt)
sales['bedrooms_square'] = sales['bedrooms']*sales['bedrooms']
sales['floors_square'] = sales['floors']*sales['floors']

testing['sqft_living_sqrt'] = testing['sqft_living'].apply(sqrt)
testing['sqft_lot_sqrt'] = testing['sqft_lot'].apply(sqrt)
testing['bedrooms_square'] = testing['bedrooms']*testing['bedrooms']
testing['floors_square'] = testing['floors']*testing['floors']

training['sqft_living_sqrt'] = training['sqft_living'].apply(sqrt)
training['sqft_lot_sqrt'] = training['sqft_lot'].apply(sqrt)
training['bedrooms_square'] = training['bedrooms']*training['bedrooms']
training['floors_square'] = training['floors']*training['floors']

validation['sqft_living_sqrt'] = validation['sqft_living'].apply(sqrt)
validation['sqft_lot_sqrt'] = validation['sqft_lot'].apply(sqrt)
validation['bedrooms_square'] = validation['bedrooms']*validation['bedrooms']
validation['floors_square'] = validation['floors']*validation['floors']

# Linear Regression Model with Lasso Rugularization
all_features = ['bedrooms', 'bedrooms_square',
            'bathrooms',
            'sqft_living', 'sqft_living_sqrt',
            'sqft_lot', 'sqft_lot_sqrt',
            'floors', 'floors_square',
            'waterfront', 'view', 'condition', 'grade',
            'sqft_above',
            'sqft_basement',
            'yr_built', 'yr_renovated']

model_all = linear_model.Lasso(alpha=5e2, normalize=True) # set parameters
model_all.fit(sales[all_features], sales['price']) # learn weights
coeffs = pd.DataFrame(list(zip(sales[all_features],model_all.coef_)),columns=['features', 'estimated coefficients'])
pos_coeffs = coeffs[coeffs['estimated coefficients']!=0]
print("Positive Coefficients: \n", pos_coeffs)

# Cross Validation for finding lambda
validation_rss = {}

for l1_penalty in np.logspace(1,7, num=13):
    model = linear_model.Lasso(l1_penalty, normalize = True)
    model.fit(training[all_features],training['price'])
    predictions = model.predict(validation[all_features])
    residuals = validation['price'] - predictions
    RSS = sum(residuals ** 2)
    validation_rss[l1_penalty] = RSS

optimal_l1_penalty = min(validation_rss.items(), key = lambda x: x[1])[0]
print() 
print("Optimal L1 Penalty: ", optimal_l1_penalty)

# Applying optimal l1 penalty on Test set'
model_best = linear_model.Lasso(alpha=optimal_l1_penalty,normalize = True, max_iter = 10000)
model_best.fit(testing[all_features], testing['price'])
coeffs = pd.DataFrame(list(zip(testing[all_features],model_best.coef_)),columns=['features','estimated coefficients'])
pos_coeffs = coeffs[coeffs['estimated coefficients']!=0]
print("Positive Coefficients (Test Set): \n", pos_coeffs)
print("Number of non-zero coeffs: ", np.count_nonzero(model_best.coef_) + np.count_nonzero(model_best.intercept_))


# Limit the number of nonzero coeffs
max_nonzeros = 7
l1_penalty_wide = np.logspace(1,4,num=20) # Wide Range

coeffs_dict = {}
for l1_penalty in l1_penalty_wide:
    model = linear_model.Lasso(alpha = l1_penalty, normalize = True)
    model.fit(training[all_features],training['price'])
    nnz = np.count_nonzero(model.coef_) + np.count_nonzero(model.intercept_)
    coeffs_dict[l1_penalty] = nnz

print("L1 penalty / # of Coefficients \n", coeffs_dict)
l1_penalty_min = 127.42749857031335
l1_penalty_max = 183.29807108324357
print("L1_penalty_min: ", l1_penalty_min)
print("L1_penalty_max: ", l1_penalty_max)

# Exploring Narrow Range
validation_rss = {}
l1_penalty_narrow = np.linspace(l1_penalty_min,l1_penalty_max, num=20)
for l1_penalty in l1_penalty_narrow:
    model = linear_model.Lasso(alpha = l1_penalty, normalize = True)
    model.fit(training[all_features],training['price'])
    predictions = model.predict(validation[all_features])
    residuals = validation['price'] - predictions
    RSS = sum(residuals ** 2)
    validation_rss[l1_penalty] = RSS, np.count_nonzero(model.coef_) + np.count_nonzero(model.intercept_)

# Find the model that the lowest RSS on the Validation set and has sparsity equal to max_nonzero
print()
bestRSS = np.inf
for i,j in validation_rss.items():
    if (j[0] < bestRSS) and (j[1] == max_nonzeros):
        bestRSS = j[0]
        bestl1 = i

print("BEST RSS: %.4g | BEST L1 PENALTY: %.4g" %(bestRSS, bestl1))

# Apply best L1 penalty
model_best = linear_model.Lasso(alpha=bestl1,normalize = True, max_iter = 10000)
model_best.fit(training[all_features], training['price'])
coeffs = pd.DataFrame(list(zip(training[all_features],model_best.coef_)),columns=['features','estimated coefficients'])
pos_coeffs = coeffs[coeffs['estimated coefficients']!=0]
print("Positive Coefficients (Test Set): \n", pos_coeffs)
print("Number of non-zero coeffs: ", np.count_nonzero(model_best.coef_) + np.count_nonzero(model_best.intercept_))