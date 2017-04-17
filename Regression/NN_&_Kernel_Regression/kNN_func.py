# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 17:25:52 2017

@author: Jihoon_Kim
"""

import numpy as np

def normalize_features(features):
    norms = np.linalg.norm(features, axis = 0)
    X_normalized = features / norms
    return(X_normalized, norms)

def compute_distances(train_matrix, query_vector):
    diff = train_matrix - query_vector
    distances = np.sqrt(np.sum(diff**2, axis=1))
    return distances

def k_nearest_neighbors(k, feature_train, features_query):
    distances = compute_distances(feature_train, features_query)
    neighbors = np.argsort(distances,axis=0)[:k]
    return neighbors

def predict_output_of_query(k, features_train, output_train, features_query):
    k_neighbors = k_nearest_neighbors(k,features_train,features_query)
    prediction = np.mean(output_train[k_neighbors])
    return prediction

def predict_output(k, features_train, output_train, features_query):
    nrow = features_query.shape[0]
    predictions = []
    for i in range(nrow):
        prediction = predict_output_of_query(k,features_train, output_train, features_query[i])
        predictions.append(prediction)
    return predictions