# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 17:19:12 2017

@author: Jihoon_Kim
"""

# Import module
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kNN_func import normalize_features
from kNN_func import compute_distances
from kNN_func import k_nearest_neighbors
from kNN_func import predict_output_of_query
from kNN_func import predict_output

# Import Data Type
dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 
              'sqft_living15':float, 'grade':int, 'yr_renovated':int, 
              'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 
              'sqft_lot15':float, 'sqft_living':float, 'floors':str, 
              'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 
              'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

# Load Data from CSV files
sales = pd.read_csv("kc_house_data_small.csv", dtype = dtype_dict)
train_data = pd.read_csv("kc_house_data_small_train.csv", dtype = dtype_dict)
test_data = pd.read_csv("kc_house_data_small_test.csv", dtype = dtype_dict)
valid_data = pd.read_csv("kc_house_data_validation.csv", dtype = dtype_dict)


# In the dataset, 'floors' was defined with type string, 
# so we'll convert them to int, before using it below
sales['floors'] = sales['floors'].astype(float)
sales['constant'] = 1
train_data['floors'] = train_data['floors'].astype(float)
train_data['constant'] = 1
test_data['floors'] = test_data['floors'].astype(float)
test_data['constant'] = 1
valid_data['floors'] = valid_data['floors'].astype(float)
valid_data['constant'] = 1

# Extract and Normalize
feature_list = ['constant',
                'bedrooms',  
                'bathrooms',  
                'sqft_living',  
                'sqft_lot',  
                'floors',
                'waterfront',  
                'view',  
                'condition',  
                'grade',  
                'sqft_above',  
                'sqft_basement',
                'yr_built',  
                'yr_renovated',  
                'lat',  
                'long',  
                'sqft_living15',  
                'sqft_lot15']

features_train = train_data[feature_list].as_matrix()
output_train = train_data['price'].as_matrix()
features_test = test_data[feature_list].as_matrix()
output_test = test_data['price'].as_matrix()
features_valid = valid_data[feature_list].as_matrix()
output_valid = valid_data['price'].as_matrix()

features_train, norms = normalize_features(features_train) # normalize training set features (columns)
features_test = features_test / norms # normalize test set by training set norms
features_valid = features_valid / norms # normalize validation set by training set norms

# Compute a single Distance
print("Query House: ", features_test[0])
print("10th House: ", features_train[9])
euclidian_distance = np.sqrt(np.sum((features_test[0] - features_train[9])**2))
print("Euclidian Distance of Query House and 10th House in Train set: ", euclidian_distance)

# Compute multiple distances
print("Distance from first 10 data: ")
dist_dict = {}
for i in range(0,10):
    dist_dict[i] = np.sqrt(np.sum((features_train[i] - features_test[0])**2))
    print (i, np.sqrt(np.sum((features_train[i] - features_test[0])**2)))
    
closest = min(dist_dict.items(), key=lambda x: x[1])
print("Closest data: ", closest)

print()
for i in range(3):
    print(features_train[i]-features_test[0])
    # should print 3 vectors of length 18
    
print(features_train[0:3] - features_test[0])

# Perform 1-nearest neighbor regression
diff = features_train - features_test[0]
distances = np.sqrt(np.sum(diff**2, axis=1))
print("Distance from 15th house in Train set: ", distances[100]) # Euclidean distance between the query house and the 101th training house

print("===============================")
print("Distance form 3rd House: ")
third_house_distance = compute_distances(features_train, features_test[2])
print("Closest Distance: ", third_house_distance[382])
print("Closest Price: ", output_train[382])

# Perform k-nearest neighbor regression

# Take the query house to be third house of the test set (features_test[2]). What are the indices of the 4 training houses closest to the query house?
print("Closest House ID: ", k_nearest_neighbors(4, features_train, features_test[2]))

# Make a single prediction by averaging k nearest neighbor outputs
print()
print("Predict the value of the query house using k-nearest neighbors with k=4 : ", predict_output_of_query(4, features_train, output_train, features_test[2]))

# Make multiple predictions
# predictions for the first 10 houses in the test set, using k=10.
print()
print("Predictions for the first 10 houses in the test set, using k= 10:" , predict_output(10,features_train,output_train,features_test[0:10]))

# Choosing the best value of k using a validation set
rss_all = {}
for k in range(1,16):    
    predict_value = predict_output(k, features_train, output_train, features_valid)
    residual = (output_valid - predict_value)
    rss = sum(residual**2)
    rss_all[k] = rss

best_k = min(rss_all.items(), key = lambda x: x[1])[0]
print("Best K: ", best_k)
# To visualize the performance as a function of k, plot the RSS on the VALIDATION set for each considered k value:
plt.plot(list(rss_all.keys()), list(rss_all.values()),'bo-')
plt.show()

predict_value_test = predict_output(best_k, features_train, output_train, features_test)
residual = (output_test - predict_value_test)
RSS = sum(residual**2)
print("RSS on Test Set: ", RSS)