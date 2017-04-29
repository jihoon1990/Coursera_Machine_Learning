# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 18:29:16 2017

@author: Jihoon_Kim
"""

# Gaussian Mixture Model to Text Data

# import module
import numpy as np
import pandas as pd

from gmm_with_em_text_func import load_sparse_csr
from gmm_with_em_text_func import diag
from gmm_with_em_text_func import EM_for_high_dimension
from gmm_with_em_text_func import visualize_EM_clusters

from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans

from sklearn.preprocessing import normalize

# Load Data
wiki = pd.read_csv("people_wiki.csv").head(5000)

# tf idf vector
tf_idf = load_sparse_csr('tf_idf.npz')
tf_idf = normalize(tf_idf)

# load index to word map
idx_to_word = pd.read_json("map_index_to_word.json", typ='series')
idx_to_word = idx_to_word.sort_values()



np.random.seed(5)
num_clusters = 25

# Use scikit-learn's k-means to simplify workflow
kmeans_model = KMeans(n_clusters=num_clusters, n_init=5, max_iter=400, random_state=1, n_jobs=-1)
kmeans_model.fit(tf_idf)
centroids, cluster_assignment = kmeans_model.cluster_centers_, kmeans_model.labels_

means = [centroid for centroid in centroids]

num_docs = tf_idf.shape[0]
weights = []
for i in range(num_clusters):
    # Compute the number of data points assigned to cluster i:
    num_assigned = cluster_assignment[cluster_assignment == i].shape[0] # YOUR CODE HERE
    w = float(num_assigned) / num_docs
    weights.append(w)
    
    
covs = []
for i in range(num_clusters):
    member_rows = tf_idf[cluster_assignment==i]
    cov = (member_rows.multiply(member_rows) - 2*member_rows.dot(diag(means[i]))).sum(axis=0).A1 / member_rows.shape[0] \
          + means[i]**2
    cov[cov < 1e-8] = 1e-8
    covs.append(cov)
    
out = EM_for_high_dimension(tf_idf, means, covs, weights, cov_smoothing=1e-10)
print(out['loglik']) # print history of log-likelihood over time

# Interpret clusters

'''By EM'''
visualize_EM_clusters(tf_idf, out['means'], out['covs'], idx_to_word)


# Comparing to random initialization
np.random.seed(5)
num_clusters = len(means)
num_docs, num_words = tf_idf.shape

random_means = []
random_covs = []
random_weights = []

for k in range(num_clusters):
    
    # Create a numpy array of length num_words with random normally distributed values.
    # Use the standard univariate normal distribution (mean 0, variance 1).
    # YOUR CODE HERE
    mean = np.random.normal(0, 1, num_words)
    
    # Create a numpy array of length num_words with random values uniformly distributed between 1 and 5.
    # YOUR CODE HERE
    cov = np.random.uniform(1,6,num_words)

    # Initially give each cluster equal weight.
    # YOUR CODE HERE
    weight = 1
    
    random_means.append(mean)
    random_covs.append(cov)
    random_weights.append(weight)
    
# Question
out_random_init = EM_for_high_dimension(tf_idf, random_means, random_covs, random_weights, cov_smoothing=1e-5)
out_random_init['loglik']

# Question
visualize_EM_clusters(tf_idf, out_random_init['means'], out_random_init['covs'], idx_to_word)
