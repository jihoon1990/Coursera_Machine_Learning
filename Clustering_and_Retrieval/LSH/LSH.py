# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 16:21:52 2017

@author: Jihoon_Kim
"""

# Locality Sensitive Hashing
# import module
import pandas as pd
import numpy as np                             
import matplotlib.pyplot as plt   
import time                             
from itertools import combinations
from LSH_func import load_sparse_csr
from LSH_func import generate_random_vectors
from LSH_func import train_lsh
from LSH_func import cosine_distance
from LSH_func import query
from LSH_func import brute_force_query

# Load Data
wiki = pd.read_csv("people_wiki.csv")

# word count vector
corpus = load_sparse_csr('people_wiki_tf_idf.npz')
# load index to word map
idx_to_word = pd.read_json("people_wiki_map_index_to_word.json", typ='series')
idx_to_word = idx_to_word.sort_values()

# Generate 16 random vectors of dimension 547979
np.random.seed(0)
random_vectors = generate_random_vectors(num_vector=16, dim=547979)
print(random_vectors.shape)

doc = corpus[0, :] # vector of tf-idf values for document 0
print(doc.dot(random_vectors[:, 0]) >= 0) # True if positive sign; False if negative sign
print(doc.dot(random_vectors[:, 1]) >= 0) # True if positive sign; False if negative sign

print(doc.dot(random_vectors) >= 0) # should return an array of 16 True/False
print(np.array(doc.dot(random_vectors) >= 0, dtype=int)) # display index bits in 0/1's

doc = corpus[0, :]  # first document
index_bits = (doc.dot(random_vectors) >= 0)
powers_of_two = (1 << np.arange(15, -1, -1))
print("Index bits: ", index_bits)
print("Power of two: ", powers_of_two)       # [32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
print(index_bits.dot(powers_of_two))
index_bits = corpus.dot(random_vectors) >= 0
print(index_bits.dot(powers_of_two))

# train LSH
model = train_lsh(corpus, num_vector=16, seed=143)
table = model['table']

# Inspect bins
print("BIN inspection: \n", wiki[wiki['name'] == 'Barack Obama'])


print("Joe Biden: \n", wiki[wiki['name'] == 'Joe Biden'])
# bin_index of Joe Biden
print("Bin index of Joe Biden: \n", np.array(model['bin_index_bits'][24478], dtype=int))# list of 0/1's
# bit representations of the bins containing Joe Biden
print("Bit representations of the bins containing Joe Biden: ", model['bin_indices'][24478]) # integer format
print(model['bin_index_bits'][35817] == model['bin_index_bits'][24478])
print(sum(model['bin_index_bits'][35817] == model['bin_index_bits'][24478]))

# Compare the result with a former British diplomat Wynn Normington Hugh-Jones
print("BIN inspection: \n", wiki[wiki['name']=='Wynn Normington Hugh-Jones'])
print(np.array(model['bin_index_bits'][22745], dtype=int)) # list of 0/1's
print(model['bin_index_bits'][35817] == model['bin_index_bits'][22745])


model['table'][model['bin_indices'][35817]]
doc_ids = list(model['table'][model['bin_indices'][35817]])
doc_ids.remove(35817) # display documents other than Obama
docs = wiki.loc[doc_ids]
print(docs)

obama_tf_idf = corpus[35817,:]
biden_tf_idf = corpus[24478,:]


# Cosine distance from Obama
print('=========Cosine distance from Barack Obama========')
print('Barack Obama - {0:24s}: {1:f}'.format('Joe Biden', cosine_distance(obama_tf_idf, biden_tf_idf)))

for doc_id in doc_ids:
    doc_tf_idf = corpus[doc_id,:]
    print('Barack Obama - {0:24s}: {1:f}'.format(wiki.loc[doc_id]['name'],
                                                 cosine_distance(obama_tf_idf, doc_tf_idf)))
    
# Query the LSH model

num_vector = 16
search_radius = 3

for diff in combinations(range(num_vector), search_radius):
    print(diff)
    
# Let's try it out with Obama
LSH_obama = wiki['name'].loc[query(corpus[35817,:], model, k=10, max_search_radius=3)[0].index].to_frame()
LSH_obama['distance'] = query(corpus[35817,:], model, k=10, max_search_radius=3)[0]
print("Query: Obama \n", LSH_obama)

# Experimenting with your LSH implementation

print(wiki[wiki['name']=='Barack Obama'])

num_candidates_history = []
query_time_history = []
max_distance_from_query_history = []
min_distance_from_query_history = []
average_distance_from_query_history = []

for max_search_radius in range(17):
    start=time.time()
    result, num_candidates = query(corpus[35817,:], model, k=10,
                                   max_search_radius=max_search_radius)
    end=time.time()
    query_time = end-start
    
    print('Radius:', max_search_radius)
    LSH = wiki['name'].loc[result.index].to_frame()
    LSH['distance'] = result
    print(LSH)
    
    average_distance_from_query = result['distance'][1:].mean()
    max_distance_from_query = result['distance'][1:].max()
    min_distance_from_query = result['distance'][1:].min()
    
    num_candidates_history.append(num_candidates)
    query_time_history.append(query_time)
    average_distance_from_query_history.append(average_distance_from_query)
    max_distance_from_query_history.append(max_distance_from_query)
    min_distance_from_query_history.append(min_distance_from_query)

# Plot
plt.figure(figsize=(7,4.5))
plt.plot(num_candidates_history, linewidth=4)
plt.xlabel('Search radius')
plt.ylabel('# of documents searched')
plt.rcParams.update({'font.size':16})
plt.tight_layout()

plt.figure(figsize=(7,4.5))
plt.plot(query_time_history, linewidth=4)
plt.xlabel('Search radius')
plt.ylabel('Query time (seconds)')
plt.rcParams.update({'font.size':16})
plt.tight_layout()

plt.figure(figsize=(7,4.5))
plt.plot(average_distance_from_query_history, linewidth=4, label='Average of 10 neighbors')
plt.plot(max_distance_from_query_history, linewidth=4, label='Farthest of 10 neighbors')
plt.plot(min_distance_from_query_history, linewidth=4, label='Closest of 10 neighbors')
plt.xlabel('Search radius')
plt.ylabel('Cosine distance of neighbors')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size':16})
plt.tight_layout()


#
for i, v in enumerate(average_distance_from_query_history):
    if v <= 0.78:
        print(i, v)
        
# Quality metrics for neighbors

max_radius = 17
precision = {i:[] for i in range(max_radius)}
average_distance  = {i:[] for i in range(max_radius)}
query_time  = {i:[] for i in range(max_radius)}

np.random.seed(0)
num_queries = 10
for i, ix in enumerate(np.random.choice(corpus.shape[0], num_queries, replace=False)):
    print('%s / %s' % (i, num_queries))
    ground_truth = set(brute_force_query(corpus[ix,:], corpus, k=25).index)
    # Get the set of 25 true nearest neighbors
    
    for r in range(1,max_radius):
        start = time.time()
        result, num_candidates = query(corpus[ix,:], model, k=10, max_search_radius=r)
        end = time.time()

        query_time[r].append(end-start)
        # precision = (# of neighbors both in result and ground_truth)/10.0
        precision[r].append(len(set(result.index) & ground_truth)/10.0)
        average_distance[r].append(result['distance'][1:].mean())
        
# Plot
plt.figure(figsize=(7,4.5))
plt.plot(range(1,17), [np.mean(average_distance[i]) for i in range(1,17)], linewidth=4, label='Average over 10 neighbors')
plt.xlabel('Search radius')
plt.ylabel('Cosine distance')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size':16})
plt.tight_layout()

plt.figure(figsize=(7,4.5))
plt.plot(range(1,17), [np.mean(precision[i]) for i in range(1,17)], linewidth=4, label='Precison@10')
plt.xlabel('Search radius')
plt.ylabel('Precision')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size':16})
plt.tight_layout()

plt.figure(figsize=(7,4.5))
plt.plot(range(1,17), [np.mean(query_time[i]) for i in range(1,17)], linewidth=4, label='Query time')
plt.xlabel('Search radius')
plt.ylabel('Query time (seconds)')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size':16})
plt.tight_layout()


# Effect of number of random vectors
precision = {i:[] for i in range(5,20)}
average_distance  = {i:[] for i in range(5,20)}
query_time = {i:[] for i in range(5,20)}
num_candidates_history = {i:[] for i in range(5,20)}
ground_truth = {}

np.random.seed(0)
num_queries = 10
docs = np.random.choice(corpus.shape[0], num_queries, replace=False)

for i, ix in enumerate(docs):
    ground_truth[ix] = set(brute_force_query(corpus[ix,:], corpus, k=25).index)
    # Get the set of 25 true nearest neighbors

for num_vector in range(5,20):
    print('num_vector = %s' % (num_vector))
    model = train_lsh(corpus, num_vector, seed=143)
    
    for i, ix in enumerate(docs):
        start = time.time()
        result, num_candidates = query(corpus[ix,:], model, k=10, max_search_radius=3)
        end = time.time()
        
        query_time[num_vector].append(end-start)
        precision[num_vector].append(len(set(result.index) & ground_truth[ix])/10.0)
        average_distance[num_vector].append(result['distance'][1:].mean())
        num_candidates_history[num_vector].append(num_candidates)

# Plot

plt.figure(figsize=(7,4.5))
plt.plot(range(5,20), [np.mean(average_distance[i]) for i in range(5,20)], linewidth=4, label='Average over 10 neighbors')
plt.xlabel('# of random vectors')
plt.ylabel('Cosine distance')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size':16})
plt.tight_layout()

plt.figure(figsize=(7,4.5))
plt.plot(range(5,20), [np.mean(precision[i]) for i in range(5,20)], linewidth=4, label='Precison@10')
plt.xlabel('# of random vectors')
plt.ylabel('Precision')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size':16})
plt.tight_layout()

plt.figure(figsize=(7,4.5))
plt.plot(range(5,20), [np.mean(query_time[i]) for i in range(5,20)], linewidth=4, label='Query time (seconds)')
plt.xlabel('# of random vectors')
plt.ylabel('Query time (seconds)')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size':16})
plt.tight_layout()

plt.figure(figsize=(7,4.5))
plt.plot(range(5,20), [np.mean(num_candidates_history[i]) for i in range(5,20)], linewidth=4,
         label='# of documents searched')
plt.xlabel('# of random vectors')
plt.ylabel('# of documents searched')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size':16})
plt.tight_layout()