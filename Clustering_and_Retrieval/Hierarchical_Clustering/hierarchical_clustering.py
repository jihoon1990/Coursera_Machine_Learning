# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 12:08:06 2017

@author: Jihoon_Kim
"""

# Hierarchical Clustering

import pandas as pd

from hierarchical_clustering_func import load_sparse_csr
from hierarchical_clustering_func import bipartition
from hierarchical_clustering_func import display_single_tf_idf_cluster
from sklearn.preprocessing import normalize

# Load Data
wiki = pd.read_csv("people_wiki.csv")

# tf idf vector
tf_idf = load_sparse_csr('people_wiki_tf_idf.npz')
tf_idf = normalize(tf_idf)

# load index to word map
idx_to_word = pd.read_json("people_wiki_map_index_to_word.json", typ='series')

wiki_data = {'matrix': tf_idf, 'dataframe': wiki} # no 'centroid' for the root cluster

left_child, right_child = bipartition(wiki_data, maxiter=100, num_runs=6, seed=1)

# Explore left & right child
print(left_child)
print(right_child)

# Display two child clusters
display_single_tf_idf_cluster(left_child, idx_to_word)
display_single_tf_idf_cluster(right_child, idx_to_word)

# Perform recursive bipartitioning
non_athletes = left_child
athletes = right_child

# Bipartition the cluster of athletes
left_child_athletes, right_child_athletes = bipartition(athletes, maxiter=100, num_runs=6, seed=1)

# left child cluster mainly consists of baseball players:
display_single_tf_idf_cluster(left_child_athletes, idx_to_word)

# right child cluster mainly consists of football players:
display_single_tf_idf_cluster(right_child_athletes, idx_to_word)

# Let's give the clusters aliases as well:

baseball            = left_child_athletes
ice_hockey_football = right_child_athletes

# Cluster of ice hockey players and football players
left_child_ihs, right_child_ihs = bipartition(ice_hockey_football, maxiter=100, num_runs=6, seed=1)
display_single_tf_idf_cluster(left_child_ihs, idx_to_word)
display_single_tf_idf_cluster(right_child_ihs, idx_to_word)

# Cluster of non-athletes.
# Bipartition the cluster of non-athletes
left_child_non_athletes, right_child_non_athletes = bipartition(non_athletes, maxiter=100, num_runs=6, seed=1)
display_single_tf_idf_cluster(left_child_non_athletes, idx_to_word)
display_single_tf_idf_cluster(right_child_non_athletes, idx_to_word)

male_non_athletes = left_child_non_athletes
female_non_athletes = right_child_non_athletes

left_child_male, right_child_male = bipartition(male_non_athletes, maxiter=100, num_runs=6, seed=1)
display_single_tf_idf_cluster(left_child_male, idx_to_word)
display_single_tf_idf_cluster(right_child_male, idx_to_word)

left_child_female, right_child_female = bipartition(female_non_athletes, maxiter=100, num_runs=6, seed=1)
display_single_tf_idf_cluster(left_child_female, idx_to_word)
display_single_tf_idf_cluster(right_child_female, idx_to_word)