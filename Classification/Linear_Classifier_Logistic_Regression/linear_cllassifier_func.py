# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 13:02:21 2017

@author: Jihoon_Kim
"""
import math

def calculate_probability(scores):
    """ Calculate the probability predictions from the scores.
    """
    prob = []
    for score in scores:
        pred =  1 / (1 + math.exp(-score))
        prob.append(pred)
    return prob