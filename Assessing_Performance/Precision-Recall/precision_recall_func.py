# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 17:02:42 2017

@author: Jihoon_Kim
"""
import pandas as pd
import matplotlib.pyplot as plt

def apply_threshold(probabilities, threshold):
    ### YOUR CODE GOES HERE
    # +1 if >= threshold and -1 otherwise.
    ans = pd.Series([+1 if x >= threshold else -1 for x in probabilities])
    return ans

def plot_pr_curve(precision, recall, title):
    plt.rcParams['figure.figsize'] = 7, 5
    plt.locator_params(axis = 'x', nbins = 5)
    plt.plot(precision, recall, 'b-', linewidth=4.0, color = '#B0017F')
    plt.title(title)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.rcParams.update({'font.size': 16})