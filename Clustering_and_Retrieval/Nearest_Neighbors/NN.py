# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 06:41:19 2017

@author: Jihoon_Kim
"""


# Nearest Neighbors

# import module
import matplotlib.pyplot as plt
import pandas as pd

from NN_func import load_sparse_csr
from NN_func import word_count_by_name
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances

# Load Data
wiki = pd.read_csv("people_wiki.csv")
wiki['length'] = wiki.apply(lambda x: len(x['text']),axis=1)
# word count vector
word_count = load_sparse_csr('people_wiki_word_count.npz')
# load index to word map
idx_to_word = pd.read_json("people_wiki_map_index_to_word.json", typ='series')
idx_to_word = idx_to_word.sort_values()

# NN model
model = NearestNeighbors(metric='euclidean', algorithm = 'brute')
model.fit(word_count)

# row number of Obama's article
print("Obama info: ", wiki[wiki['name']=='Barack Obama'])
distances, indices = model.kneighbors(word_count_by_name(wiki,word_count,'Barack Obama'), n_neighbors=10)
nn_obama = wiki.loc[indices[0]]
nn_obama['distance'] = distances[0]
print("10 Nearest neightbor of Barack Obama: \n", nn_obama[['name','distance']])

# Barack Obama word count table
obama_words = pd.DataFrame(word_count_by_name(wiki,word_count,'Barack Obama').toarray()[0],index=idx_to_word.index,columns=['count'])
obama_words = obama_words.sort_values(by='count',ascending=False)
print("Obama words: \n", obama_words.head(10))

# Francisco Barrio's pages.
# row number of Francisco Barrio's article
wiki[wiki['name']=='Francisco Barrio']

distances, indices = model.kneighbors(word_count_by_name(wiki,word_count,'Francisco Barrio'), n_neighbors=10)
nn_barrio = wiki.loc[indices[0]]
nn_barrio['distance'] = distances[0]
nn_barrio[['name','distance']]
print("10 Nearest neightbor of Barrio: \n", nn_barrio[['name','distance']])

barrio_words = pd.DataFrame(word_count_by_name(wiki,word_count,'Francisco Barrio').toarray()[0],index=idx_to_word.index,columns=['count'])
barrio_words = barrio_words.sort_values(by='count',ascending=False)
print("Barrio words: \n", barrio_words.head(10))

# Combined Word
combined_words = obama_words.join(barrio_words,rsuffix='.1')
# Rename Columns
combined_words = combined_words.rename(columns={'count':'Obama','count.1':'Barrio'})
# Sort by obama
combined_words = combined_words.sort_values(by='Obama', ascending = False)
print("Combined words: \n", combined_words)

# Distance
print("Distance between Obama and George W. Bush: \n", euclidean_distances(word_count_by_name(wiki,word_count,'Barack Obama'),word_count_by_name(wiki,word_count,'George W. Bush')))
print("Distance between Obama and Joe Biden: \n", euclidean_distances(word_count_by_name(wiki,word_count,'Barack Obama'),word_count_by_name(wiki,word_count,'Joe Biden')))
print("Distance between George W. Bush and Joe Biden: \n", euclidean_distances(word_count_by_name(wiki,word_count,'Joe Biden'),word_count_by_name(wiki,word_count,'George W. Bush')))

# TF-IDF
tf_idf = load_sparse_csr('people_wiki_tf_idf.npz')

# NN model
model = NearestNeighbors(metric='euclidean', algorithm = 'brute')
model.fit(tf_idf)

# row number of Obama's article
print("Obama info: ", wiki[wiki['name']=='Barack Obama'])
distances, indices = model.kneighbors(word_count_by_name(wiki,tf_idf,'Barack Obama'), n_neighbors=100)
nn_obama = wiki.loc[indices[0]]
nn_obama['distance'] = distances[0]
print("10 Nearest neightbor of Barack Obama: \n", nn_obama[['name','distance']])

# Barack Obama word count table
obama_words = pd.DataFrame(word_count_by_name(wiki,tf_idf,'Barack Obama').toarray()[0],index=idx_to_word.index,columns=['weight'])
obama_words = obama_words.sort_values(by='weight',ascending=False)
print("Obama words: \n", obama_words.head(10))

# Schiliro Obama word count table
schiliro_words = pd.DataFrame(word_count_by_name(wiki,tf_idf,'Phil Schiliro').toarray()[0],index=idx_to_word.index,columns=['weight'])
schiliro_words = schiliro_words.sort_values(by='weight',ascending=False)
print("Schiliro words: \n", schiliro_words.head(10))

# combined words
combined_words = obama_words.join(schiliro_words,rsuffix='.1')
# Rename Columns
combined_words = combined_words.rename(columns={'weight':'Obama','weight.1':'Barrio'})
# Sort by obama
combined_words = combined_words.sort_values(by='Obama', ascending = False)
print("Combined words: \n", combined_words)

# Euclidean Distance
# Obama and Bush
print("Distance between Obama and George W. Bush: \n", euclidean_distances(word_count_by_name(wiki,tf_idf,'Barack Obama'),word_count_by_name(wiki,tf_idf,'George W. Bush')))
print("Distance between Obama and Joe Biden: \n", euclidean_distances(word_count_by_name(wiki,tf_idf,'Barack Obama'),word_count_by_name(wiki,tf_idf,'Joe Biden')))
print("Distance between George W. Bush and Joe Biden: \n", euclidean_distances(word_count_by_name(wiki,tf_idf,'Joe Biden'),word_count_by_name(wiki,tf_idf,'George W. Bush')))

# length of document
plt.figure(figsize=(10.5,4.5))
plt.hist(wiki['length'], 50, color='k', edgecolor='None', histtype='stepfilled', normed=True,
         label='Entire Wikipedia', zorder=3, alpha=0.8)
plt.hist(nn_obama['length'], 50, color='r', edgecolor='None', histtype='stepfilled', normed=True,
         label='100 NNs of Obama (Euclidean)', zorder=10, alpha=0.8)
plt.axvline(x=wiki[wiki['name']=='Barack Obama']['length'].values[0], color='k', linestyle='--', linewidth=4,
           label='Length of Barack Obama', zorder=2)
plt.axvline(x=wiki[wiki['name']=='Joe Biden']['length'].values[0], color='g', linestyle='--', linewidth=4,
           label='Length of Joe Biden', zorder=1)
plt.axis([1000, 5500, 0, 0.004])

plt.legend(loc='best', prop={'size':15})
plt.title('Distribution of document length')
plt.xlabel('# of words')
plt.ylabel('Percentage')
plt.rcParams.update({'font.size':16})
plt.tight_layout()

# Now, apply cosine distance
# cosine distance model
model_cosine= NearestNeighbors(metric='cosine', algorithm = 'brute')
model_cosine.fit(tf_idf)
# row number of Obama's article
print("Obama info: ", wiki[wiki['name']=='Barack Obama'])
distances, indices = model_cosine.kneighbors(word_count_by_name(wiki,tf_idf,'Barack Obama'), n_neighbors=100)
nn_obama_cosine = wiki.loc[indices[0]]
nn_obama_cosine['distance'] = distances[0]
print("10 Nearest neightbor of Barack Obama: \n", nn_obama_cosine[['name','distance','length']])

# with cosine distance
plt.figure(figsize=(10.5,4.5))
plt.hist(wiki['length'], 50, color='k', edgecolor='None', histtype='stepfilled', normed=True,
         label='Entire Wikipedia', zorder=3, alpha=0.8)
plt.hist(nn_obama['length'], 50, color='r', edgecolor='None', histtype='stepfilled', normed=True,
         label='100 NNs of Obama (Euclidean)', zorder=10, alpha=0.8)
plt.hist(nn_obama_cosine['length'], 50, color='b', edgecolor='None', histtype='stepfilled', normed=True,
         label='100 NNs of Obama (cosine)', zorder=11, alpha=0.8)
plt.axvline(x=wiki[wiki['name']=='Barack Obama']['length'].values[0], color='k', linestyle='--', linewidth=4,
           label='Length of Barack Obama', zorder=2)
plt.axvline(x=wiki[wiki['name']=='Joe Biden']['length'].values[0], color='g', linestyle='--', linewidth=4,
           label='Length of Joe Biden', zorder=1)
plt.axis([1000, 5500, 0, 0.004])
plt.legend(loc='best', prop={'size':15})
plt.title('Distribution of document length')
plt.xlabel('# of words')
plt.ylabel('Percentage')
plt.rcParams.update({'font.size': 16})
plt.tight_layout()

# Problem with cosine distances: tweets vs. long articles
tweet = {'act': 3.4597778278724887,
 'control': 3.721765211295327,
 'democratic': 3.1026721743330414,
 'governments': 4.167571323949673,
 'in': 0.0009654063501214492,
 'law': 2.4538226269605703,
 'popular': 2.764478952022998,
 'response': 4.261461747058352,
 'to': 0.04694493768179923}

word_indices = [idx_to_word[idx_to_word.index==word].values[0] for word in tweet.keys()]
tweet_tf_idf = csr_matrix((list(tweet.values()),([0]*len(word_indices), word_indices)),
                          shape=(1, tf_idf.shape[1]) )
# Now, compute the cosine distance between the Barack Obama article and this tweet:
print("Distance between Obama and tweet: ", cosine_distances(word_count_by_name(wiki,tf_idf,'Barack Obama'),tweet_tf_idf))