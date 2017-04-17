# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 13:35:55 2017

@author: Jihoon_Kim
"""
import numpy as np

def predict_output(feature_matrix, weights):
    # assume feature_matrix is a numpy matrix containing the features as columns and weights is a corresponding numpy array
    # create the predictions vector by using np.dot()
    predictions = np.dot(feature_matrix, weights)
    return(predictions)

def normalize_features(features):
    norms = np.linalg.norm(features, axis = 0)
    X_normalized = features / norms
    return(X_normalized, norms)


def in_l1range(value, penalty):
    # Return True if value is within the threshold ranges otherwise False
    # Looking for range -l1_penalty/2 <= ro <= l1_penalty/2
    return ( (value >= -penalty/2.) and (value <= penalty/2.) )

def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):
    # compute prediction
    prediction = predict_output(feature_matrix,weights)
    # compute ro[i] = SUM[ [feature_i]*(output - prediction + weight[i]*[feature_i]) ]
    ro_i = (feature_matrix[:,i] * (output - prediction + (weights[i] * feature_matrix[:,i]))).sum()
    
    if i == 0: # intercept -- do not regularize
        new_weight_i = ro_i
    elif ro_i < -l1_penalty/2.:
        new_weight_i = ro_i + l1_penalty/2.
    elif ro_i > l1_penalty/2.:
        new_weight_i = ro_i - l1_penalty/2.
    else:
        new_weight_i = 0.
    
    return new_weight_i


def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, l1_penalty, tolerance):
    n_feature = feature_matrix.shape[1]
    weights = np.array(initial_weights)
    change = np.array(initial_weights) * 0.0
    converged = False
    
    while not converged:
        # Evaluate over all features
        for idx in range(n_feature):
            # new weights for feature
            new_weight = lasso_coordinate_descent_step(idx,feature_matrix,output,weights,l1_penalty)
            # compute change in weight for feature
            change[idx] = np.abs(new_weight - weights[idx])
            # assign new weight
            weights[idx] = new_weight
            
        max_change = max(change)
        
        if max_change < tolerance:
            converged = True
            
    return weights