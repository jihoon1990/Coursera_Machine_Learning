# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 15:23:02 2017

@author: Jihoon_Kim
"""

# Gaussian Mixture Model with EM

# import package

import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.stats import multivariate_normal
from gmm_with_em_func import generate_MoG_data