# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 18:02:33 2017

@author: Jihoon_Kim
"""

# Bias-Variance Tradeoff

# Import Module
import pandas as pd
import matplotlib.pyplot as plt
from assessing_perf_func import polynomial_dataframe
from sklearn import linear_model

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

# Visualizing
sales = sales.sort_values(['sqft_living','price'])

# Making a 1 degree polynomial with sales[‘sqft_living’] as the the feature. 
# Call it ‘poly1_data’.
poly1_data = polynomial_dataframe(sales['sqft_living'], 1)
poly1_data['price'] = sales['price']

# Linear Regression Model
model1 = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
model1.fit(poly1_data.drop('price',1), poly1_data['price'])

print("Coefficients: ", model1.intercept_)
coeffs = pd.DataFrame(list(zip(poly1_data.drop('price',1).columns,model1.coef_)), columns = ['features', 'estimated coefficients'])
print(coeffs)

# Scatter plot of data (1st power of sqft)
plt.figure(0)
plt.plot(poly1_data[['power_1']], poly1_data[['price']],'.',
         poly1_data[['power_1']], model1.predict(poly1_data.drop('price',1)),'-')
plt.xlabel('Squarefeet')
plt.ylabel('Price')
plt.show()

# 2nd degree polynomial
poly2_data = polynomial_dataframe(sales['sqft_living'], 2)
poly2_data['price'] = sales['price']
model2 = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
model2.fit(poly2_data.drop('price',1), poly2_data['price'])
print()
print("Coefficients: ", model2.intercept_)
coeffs = pd.DataFrame(list(zip(poly2_data.drop('price',1).columns,model2.coef_)), columns = ['features', 'estimated coefficients'])
print(coeffs)

# Scatter plot of data (2nd power of sqft)
plt.figure(1)
plt.plot(poly2_data[['power_1']], poly2_data[['price']],'.',
         poly2_data[['power_1']], model2.predict(poly2_data.drop('price',1)),'-')
plt.xlabel('Squarefeet')
plt.ylabel('Price')
plt.show()

# 3rd degree polynomial
poly3_data = polynomial_dataframe(sales['sqft_living'], 3)
poly3_data['price'] = sales['price']
model3 = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
model3.fit(poly3_data.drop('price',1), poly3_data['price'])
print()
print("Coefficients: ", model3.intercept_)
coeffs = pd.DataFrame(list(zip(poly3_data.drop('price',1).columns,model3.coef_)), columns = ['features', 'estimated coefficients'])
print(coeffs)

# Scatter plot of data (3rd power of sqft)
plt.figure(2)
plt.plot(poly3_data[['power_1']], poly3_data[['price']],'.',
         poly3_data[['power_1']], model3.predict(poly3_data.drop('price',1)),'-')
plt.xlabel('Squarefeet')
plt.ylabel('Price')
plt.show()

# Let's try 15th degree polynomial:
# 15th degree polynomial
poly15_data = polynomial_dataframe(sales['sqft_living'], 15)
poly15_data['price'] = sales['price']
model15 = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
model15.fit(poly15_data.drop('price',1), poly15_data['price'])
print()
print("Coefficients: ", model15.intercept_)
coeffs = pd.DataFrame(list(zip(poly15_data.drop('price',1).columns,model15.coef_)), columns = ['features', 'estimated coefficients'])
print(coeffs)

# Scatter plot of data (3rd power of sqft)
plt.figure(3)
plt.plot(poly15_data[['power_1']], poly15_data[['price']],'.',
         poly15_data[['power_1']], model15.predict(poly15_data.drop('price',1)),'-')
plt.xlabel('Squarefeet')
plt.ylabel('Price')
plt.show()

# Changing the data and re-learning
# Estimate a 15th degree polynomial on all 4 sets, plot the results and view the coefficients for all four models.

# Set 1
poly_set1 = polynomial_dataframe(set1['sqft_living'], 15)
poly_set1['price'] = set1['price']
model_set1 = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
model_set1.fit(poly_set1.drop('price',1), poly_set1['price'])
print()
print("Coefficients: ", model_set1.intercept_)
coeffs = pd.DataFrame(list(zip(poly_set1.drop('price',1).columns,model_set1.coef_)), columns = ['features', 'estimated coefficients'])
print(coeffs)

# Scatter plot of set1 data
plt.figure(4)
plt.plot(poly_set1[['power_1']], poly_set1[['price']],'.',
         poly_set1[['power_1']], model_set1.predict(poly_set1.drop('price',1)),'-')
plt.xlabel('Squarefeet')
plt.ylabel('Price')
plt.title('Set 1')
plt.show()

# Set 2
poly_set2 = polynomial_dataframe(set2['sqft_living'], 15)
poly_set2['price'] = set2['price']
model_set2 = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
model_set2.fit(poly_set2.drop('price',1), poly_set2['price'])
print()
print("Coefficients: ", model_set2.intercept_)
coeffs = pd.DataFrame(list(zip(poly_set2.drop('price',1).columns,model_set2.coef_)), columns = ['features', 'estimated coefficients'])
print(coeffs)

# Scatter plot of set1 data
plt.figure(5)
plt.plot(poly_set2[['power_1']], poly_set2[['price']],'.',
         poly_set2[['power_1']], model_set2.predict(poly_set2.drop('price',1)),'-')
plt.xlabel('Squarefeet')
plt.ylabel('Price')
plt.title('Set 2')
plt.show()

# Set 3
poly_set3 = polynomial_dataframe(set3['sqft_living'], 15)
poly_set3['price'] = set3['price']
model_set3 = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
model_set3.fit(poly_set3.drop('price',1), poly_set3['price'])
print()
print("Coefficients: ", model_set3.intercept_)
coeffs = pd.DataFrame(list(zip(poly_set3.drop('price',1).columns,model_set3.coef_)), columns = ['features', 'estimated coefficients'])
print(coeffs)

# Scatter plot of set1 data
plt.figure(6)
plt.plot(poly_set3[['power_1']], poly_set3[['price']],'.',
         poly_set3[['power_1']], model_set3.predict(poly_set3.drop('price',1)),'-')
plt.xlabel('Squarefeet')
plt.ylabel('Price')
plt.title('Set 3')
plt.show()

# Set 4
poly_set4 = polynomial_dataframe(set2['sqft_living'], 15)
poly_set4['price'] = set4['price']
model_set4 = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
model_set4.fit(poly_set4.drop('price',1), poly_set4['price'])
print()
print("Coefficients: ", model_set4.intercept_)
coeffs = pd.DataFrame(list(zip(poly_set4.drop('price',1).columns,model_set4.coef_)), columns = ['features', 'estimated coefficients'])
print(coeffs)

# Scatter plot of set1 data
plt.figure(7)
plt.plot(poly_set4[['power_1']], poly_set4[['price']],'.',
         poly_set4[['power_1']], model_set4.predict(poly_set4.drop('price',1)),'-')
plt.xlabel('Squarefeet')
plt.ylabel('Price')
plt.title('Set 4')
plt.show()

# Selecting Polynomial Degree (Validation Data)
degree_rss_table =[]
for degree in range(1,16):
    data = polynomial_dataframe(train_data['sqft_living'], degree)
    data['price'] = train_data['price']
    model = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
    model.fit(data.drop('price',1), data['price'])
    validation_data = polynomial_dataframe(valid_data['sqft_living'], degree)
    val_pred = model.predict(data.drop('price',1))
    residuals = val_pred - data['price']
    RSS = (residuals**2).sum()
    
    print('Degree: %d / RSS: %.4g' % (degree, RSS))
    degree_rss_table.append((degree,RSS))
    
# Choosing the best model
print(sorted(degree_rss_table, key = lambda RSS: RSS[1]))

# Apply the model to Test Set
low_RSS_data = polynomial_dataframe(test_data['sqft_living'], degree)
low_RSS_data['price'] = test_data['price']
model = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
model.fit(low_RSS_data.drop('price',1), low_RSS_data['price'])
test_pred = model.predict(low_RSS_data.drop('price',1))
residuals = test_pred - low_RSS_data['price']
RSS = (residuals**2).sum()
print('RSS on Test data for the model of degree 6: %.4g' % RSS)