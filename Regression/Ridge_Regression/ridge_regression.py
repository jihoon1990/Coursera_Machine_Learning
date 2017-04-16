# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 12:19:31 2017

@author: Jihoon_Kim
"""

# Ridge Regression (Interpretation)
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from ridge_func import polynomial_dataframe
from ridge_func import ridge_model
from ridge_func import plot_fitted_line
from ridge_func import k_fold_cross_validation
# Import Data Type
dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 
              'sqft_living15':float, 'grade':int, 'yr_renovated':int, 
              'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 
              'sqft_lot15':float, 'sqft_living':float, 'floors':str, 
              'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 
              'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

# Load Data from CSV files
sales = pd.read_csv("kc_house_data.csv", dtype = dtype_dict)
train_data = pd.read_csv("kc_house_train_data.csv", dtype = dtype_dict)
test_data = pd.read_csv("kc_house_test_data.csv", dtype = dtype_dict)
valid_data = pd.read_csv("kc_house_valid_data.csv", dtype = dtype_dict)
set1 = pd.read_csv("kc_house_set_1_data.csv", dtype = dtype_dict)
set2 = pd.read_csv("kc_house_set_2_data.csv", dtype = dtype_dict)
set3 = pd.read_csv("kc_house_set_3_data.csv", dtype = dtype_dict)
set4 = pd.read_csv("kc_house_set_4_data.csv", dtype = dtype_dict)
train_valid_shuffled = pd.read_csv("kc_house_train_valid_shuffled.csv", dtype = dtype_dict)

# Linear Regression with small l2 penalty
sales = sales.sort_values(['sqft_living','price'])
sales_data = polynomial_dataframe(sales['sqft_living'], 11)
sales_data['price'] = sales['price']
l2_small_penalty = 1e-5
model1 = ridge_model(l2_small_penalty, sales_data.drop('price',1), sales_data['price'])
plot_fitted_line(0,sales_data,model1)

# Observe Overfitting
# Set 1
print()
poly_set1 = polynomial_dataframe(set1['sqft_living'],11)
poly_set1['price'] = set1['price']
model_set1 = ridge_model(l2_small_penalty,poly_set1.drop('price',1),poly_set1['price'])
plot_fitted_line(1,poly_set1,model_set1)

# Set 2
print()
poly_set2 = polynomial_dataframe(set2['sqft_living'],11)
poly_set2['price'] = set2['price']
model_set2 = ridge_model(l2_small_penalty,poly_set2.drop('price',1),poly_set2['price'])
plot_fitted_line(2,poly_set2,model_set2)

# Set 3
print()
poly_set3 = polynomial_dataframe(set3['sqft_living'],11)
poly_set3['price'] = set3['price']
model_set3 = ridge_model(l2_small_penalty,poly_set3.drop('price',1),poly_set3['price'])
plot_fitted_line(3,poly_set3,model_set3)

# Set 4
print()
poly_set4 = polynomial_dataframe(set4['sqft_living'],11)
poly_set4['price'] = set4['price']
model_set4 = ridge_model(l2_small_penalty,poly_set4.drop('price',1),poly_set4['price'])
plot_fitted_line(4,poly_set4,model_set4)

## Apply Ridge Regression
# Generally, whenever we see weights change so much in response to change in data, we believe the variance of our estimate to be large. Ridge regression aims to address this issue by penalizing "large" weights.
l2_new_penalty = 1e78
# Set 1
print()
poly_set1 = polynomial_dataframe(set1['sqft_living'],11)
poly_set1['price'] = set1['price']
model_set1 = ridge_model(l2_new_penalty,poly_set1.drop('price',1),poly_set1['price'])
plot_fitted_line(1,poly_set1,model_set1)

# Set 2
print()
poly_set2 = polynomial_dataframe(set2['sqft_living'],11)
poly_set2['price'] = set2['price']
model_set2 = ridge_model(l2_new_penalty,poly_set2.drop('price',1),poly_set2['price'])
plot_fitted_line(2,poly_set2,model_set2)

# Set 3
print()
poly_set3 = polynomial_dataframe(set3['sqft_living'],11)
poly_set3['price'] = set3['price']
model_set3 = ridge_model(l2_new_penalty,poly_set3.drop('price',1),poly_set3['price'])
plot_fitted_line(3,poly_set3,model_set3)

# Set 4
print()
poly_set4 = polynomial_dataframe(set4['sqft_living'],11)
poly_set4['price'] = set4['price']
model_set4 = ridge_model(l2_new_penalty,poly_set4.drop('price',1),poly_set4['price'])
plot_fitted_line(4,poly_set4,model_set4)

# Curves with L2 regularization varies much less than without regularization

# Selecting an L2 penalty via cross-validation
# Just like the polynomial degree, the L2 penalty is a "magic" parameter we need to select. We could use the validation set approach as we did in the last module, but that approach has a major disadvantage: it leaves fewer observations available for training. Cross-validation seeks to overcome this issue by using all of the training set in a smart way.
# We will implement a kind of cross-validation called k-fold cross-validation. The method gets its name because it involves dividing the training set into k segments of roughtly equal size. Similar to the validation set method, we measure the validation error with one of the segments designated as the validation set. The major difference is that we repeat the process k times as follows:
# Set aside segment 0 as the validation set, and fit a model on rest of data, and evalutate it on this validation set
# Set aside segment 1 as the validation set, and fit a model on rest of data, and evalutate it on this validation set
# ...
# Set aside segment k-1 as the validation set, and fit a model on rest of data, and evalutate it on this validation set
# After this process, we compute the average of the k validation errors, and use it as an estimate of the generalization error. Notice that all observations are used for both training and validation, as we iterate over segments of data.
# To estimate the generalization error well, it is crucial to shuffle the training data before dividing them into segments. GraphLab Create has a utility function for shuffling a given SFrame. We reserve 10% of the data as the test set and shuffle the remainder. (Make sure to use seed=1 to get consistent answer.)

poly_data = polynomial_dataframe(train_valid_shuffled['sqft_living'], 15)
poly_data['price'] = train_valid_shuffled['price']

val_err_dict = {}
for l2_penalty in np.logspace(70, 85, num=100):
    val_err = k_fold_cross_validation(10, l2_penalty, poly_data)    
    val_err_dict[l2_penalty] = val_err

l2_penalty = val_err_dict.keys()
validation_error = val_err_dict.values()

l2_table = pd.DataFrame(list(val_err_dict.items()),columns=['l2_penalty','validation_error'])
plt.plot(l2_table['l2_penalty'],l2_table['validation_error'],'k.')
plt.xscale('log')
plt.xlabel('L2 penalty')
plt.ylabel('error')

optimal_penalty = min(val_err_dict.items(), key = lambda x:x[1])
print("Optimal L2 Penalty: ", optimal_penalty)

# Linear Regression with optimal L2 penalty
sales = sales.sort_values(['sqft_living','price'])
sales_data = polynomial_dataframe(sales['sqft_living'], 11)
sales_data['price'] = sales['price']
l2_penalty_best = optimal_penalty[0]
model_best = ridge_model(l2_penalty_best, sales_data.drop('price',1), sales_data['price'])
plot_fitted_line(5,sales_data,model_best)

errors = model_best.predict(sales_data.drop('price',1))
rss = (errors*errors).sum()
print("RSS with Optimal L2 penalty on Data set: ", rss)