# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 06:44:00 2017

@author: Jihoon_Kim
"""

"""
NN_func.py
"""

import numpy as np
from scipy.sparse import csr_matrix

def load_sparse_csr(filename):
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']
    
    return csr_matrix((data, indices, indptr), shape)

def word_count_by_name(data, count_vector, name):
    wiki_id = data[data['name']==name].index[0]
    return count_vector[wiki_id]
    
    