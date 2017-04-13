# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:00:05 2017

@author: Jihoon Kim

"""
def simple_linear_regression(input_feature, output):
    # compute the sum of input_feature and output
    sum_input_feature = input_feature.sum()
    sum_output = output.sum()
    
    # compute the product of the output and the input_feature and its sum
    mult_input_feature_output = input_feature * output
    sum_output_input = mult_input_feature_output.sum()
    
    # compute the squared value of the input_feature and its sum
    input_squared = input_feature * input_feature
    sum_input_squared = input_squared.sum()
    
    # use the formula for the slope
    num_data = output.size
    slope = (sum_output_input-(sum_output*sum_input_feature/num_data))/(sum_input_squared-input_feature.sum()*input_feature.sum()/num_data)
    # use the formula for the intercept
    intercept = output.mean() - slope * input_feature.mean()
    return (intercept, slope)

def get_regression_predictions(input_feature, intercept, slope):
    # calculate the predicted values:
    predicted_output = intercept + slope*input_feature    
    return predicted_output

def get_residual_sum_of_squares(input_feature, output, intercept,slope):
    # First get the predictions
    predictions = intercept + slope*input_feature
    # then compute the residuals (since we are squaring it doesn't matter which order you subtract)
    residuals = output - predictions
    # square the residuals and add them up
    squared_residuals = residuals ** 2
    RSS = squared_residuals.sum()
    return RSS 

def inverse_regression_predictions(output, intercept, slope):
    estimated_input = (output-intercept)/slope
    return estimated_input