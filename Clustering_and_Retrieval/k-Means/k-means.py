# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 01:24:40 2017

@author: Jihoon_Kim
"""

# k-Means

## import modules
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np                                   # dense matrices
from k_means_func import load_sparse_csr
from k_means_func import get_initial_centroids
from k_means_func import assign_clusters
from k_means_func import revise_centroids
from k_means_func import compute_heterogeneity
from k_means_func import kmeans
from k_means_func import plot_heterogeneity
from k_means_func import smart_initialize
from k_means_func import plot_k_vs_heterogeneity
from k_means_func import visualize_document_clusters
from scipy.sparse import csr_matrix                  # sparse matrices
from sklearn.preprocessing import normalize          # normalizing vectors
from sklearn.metrics import pairwise_distances       # pairwise distances
import sys      
import os

# Load Data
wiki = pd.read_csv("people_wiki.csv")

# word count vector
tf_idf = load_sparse_csr('people_wiki_tf_idf.npz')

# load index to word map
map_index_to_word = pd.read_json("people_wiki_map_index_to_word.json", typ='series')
map_index_to_word = map_index_to_word.sort_values()

# normalize tf-idf
tf_idf = normalize(tf_idf)

# Get the TF-IDF vectors for documents 100 through 102.
queries = tf_idf[100:102,:]

# Assigning clusters. 
# Compute pairwise distances from every data point to each query vector.
dist = pairwise_distances(tf_idf, queries, metric='euclidean')

# Checkpoint:
    
# Students should write code here
first_3_centroids = tf_idf[:3,:]
distances = pairwise_distances(tf_idf, first_3_centroids, metric='euclidean')
dist = distances[430, 1]

'''Test cell'''
if np.allclose(dist, pairwise_distances(tf_idf[430,:], tf_idf[1,:])):
    print('Pass')
else:
    print('Check your code again')
    
# Checkpoint: 

# Students should write code here
distances = distances.copy()
closest_cluster = np.argmin(distances, axis=1)
print(closest_cluster)
print(closest_cluster.shape)

'''Test cell'''
reference = [list(row).index(min(row)) for row in distances]
if np.allclose(closest_cluster, reference):
    print('Pass')
else:
    print('Check your code again')
    
# Checkpoint: 
# Students should write code here
first_3_centroids = tf_idf[:3,:]
distances = pairwise_distances(tf_idf, first_3_centroids, metric='euclidean')
cluster_assignment = np.argmin(distances, axis=1)

if len(cluster_assignment)==59071 and \
   np.array_equal(np.bincount(cluster_assignment), np.array([23061, 10086, 25924])):
    print('Pass') # count number of data points for each cluster
else:
    print('Check your code again.')
    

# Checkpoint:
if np.allclose(assign_clusters(tf_idf[0:100:10], tf_idf[0:8:2]), np.array([0, 1, 1, 0, 0, 2, 0, 2, 2, 1])):
    print('Pass')
else:
    print('Check your code again.')
    
# Revising clusters
data = np.array([[1., 2., 0.],
                 [0., 0., 0.],
                 [2., 2., 0.]])
centroids = np.array([[0.5, 0.5, 0.],
                      [0., -0.5, 0.]])
    
cluster_assignment = assign_clusters(data, centroids)
print(cluster_assignment)  # prints [0 1 0]

print(data[cluster_assignment==1])
print(data[cluster_assignment==0])

print(data[cluster_assignment==0].mean(axis=0))

# Check point
result = revise_centroids(tf_idf[0:100:10], 3, np.array([0, 1, 1, 0, 0, 2, 0, 2, 2, 1]))
if np.allclose(result[0], np.mean(tf_idf[[0,30,40,60]].toarray(), axis=0)) and \
   np.allclose(result[1], np.mean(tf_idf[[10,20,90]].toarray(), axis=0))   and \
   np.allclose(result[2], np.mean(tf_idf[[50,70,80]].toarray(), axis=0)):
    print('Pass')
else:
    print('Check your code')
    
# Assessing convergence

# Combining into a single function

# Plotting convergence metric

k = 3
heterogeneity = []
initial_centroids = get_initial_centroids(tf_idf, k, seed=0)
centroids, cluster_assignment = kmeans(tf_idf, k, initial_centroids, maxiter=400,
                                       record_heterogeneity=heterogeneity, verbose=True)
plot_heterogeneity(heterogeneity, k)

# Beware of local minima
k = 10
heterogeneity = {}
import time
start = time.time()
for seed in [0, 20000, 40000, 60000, 80000, 100000, 120000]:
    initial_centroids = get_initial_centroids(tf_idf, k, seed)
    centroids, cluster_assignment = kmeans(tf_idf, k, initial_centroids, maxiter=400,
                                           record_heterogeneity=None, verbose=False)
    # To save time, compute heterogeneity only once in the end
    heterogeneity[seed] = compute_heterogeneity(tf_idf, k, centroids, cluster_assignment)
    print('seed={0:06d}, heterogeneity={1:.5f}'.format(seed, heterogeneity[seed]))
    sys.stdout.flush()
end = time.time()
print(end-start)

k = 10
heterogeneity_smart = {}
start = time.time()
for seed in [0, 20000, 40000, 60000, 80000, 100000, 120000]:
    initial_centroids = smart_initialize(tf_idf, k, seed)
    centroids, cluster_assignment = kmeans(tf_idf, k, initial_centroids, maxiter=400,
                                           record_heterogeneity=None, verbose=False)
    # To save time, compute heterogeneity only once in the end
    heterogeneity_smart[seed] = compute_heterogeneity(tf_idf, k, centroids, cluster_assignment)
    print('seed={0:06d}, heterogeneity={1:.5f}'.format(seed, heterogeneity_smart[seed]))
    sys.stdout.flush()
end = time.time()
print(end-start)

plt.figure(figsize=(8,5))
plt.boxplot([list(heterogeneity.values()), list(heterogeneity_smart.values())], vert=False)
plt.yticks([1, 2], ['k-means', 'k-means++'])
plt.rcParams.update({'font.size': 16})
plt.tight_layout()

# How to choose K
#def plot_k_vs_heterogeneity(k_values, heterogeneity_values):
#    plt.figure(figsize=(7,4))
#    plt.plot(k_values, heterogeneity_values, linewidth=4)
#    plt.xlabel('K')
#    plt.ylabel('Heterogeneity')
#    plt.title('K vs. Heterogeneity')
#    plt.rcParams.update({'font.size': 16})
#    plt.tight_layout()

#start = time.time()
#centroids = {}
#cluster_assignment = {}
#heterogeneity_values = []
#k_list = [2, 10, 25, 50, 100]
#seed_list = [0, 20000, 40000, 60000, 80000, 100000, 120000]

#for k in k_list:
#    heterogeneity = []
#    centroids[k], cluster_assignment[k] = kmeans_multiple_runs(tf_idf, k, maxiter=400,
#                                                               num_runs=len(seed_list),
#                                                               seed_list=seed_list,
#                                                               verbose=True)
#    score = compute_heterogeneity(tf_idf, k, centroids[k], cluster_assignment[k])
#    heterogeneity_values.append(score)

#plot_k_vs_heterogeneity(k_list, heterogeneity_values)

#end = time.time()
#print(end-start)

filename = 'kmeans-arrays.npz'

heterogeneity_values = []
k_list = [2, 10, 25, 50, 100]

if os.path.exists(filename):
    arrays = np.load(filename)
    centroids = {}
    cluster_assignment = {}
    for k in k_list:
        print(k)
        sys.stdout.flush()
        centroids[k] = arrays['centroids_{0:d}'.format(k)]
        cluster_assignment[k] = arrays['cluster_assignment_{0:d}'.format(k)]
        score = compute_heterogeneity(tf_idf, k, centroids[k], cluster_assignment[k])
        heterogeneity_values.append(score)
    
    plot_k_vs_heterogeneity(k_list, heterogeneity_values)

else:
    print('File not found. Skipping.')

visualize_document_clusters(wiki, tf_idf, centroids[2], cluster_assignment[2], 2, map_index_to_word)


k = 10
visualize_document_clusters(wiki, tf_idf, centroids[k], cluster_assignment[k], k, map_index_to_word)

np.bincount(cluster_assignment[10])


visualize_document_clusters(wiki, tf_idf, centroids[25], cluster_assignment[25], 25, map_index_to_word, display_content=False) # turn off text for brevity

k=100
visualize_document_clusters(wiki, tf_idf, centroids[k], cluster_assignment[k], k, map_index_to_word, display_content=False) # turn off text for brevity -- turn it on if you are curious ;)
