#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 14:30:43 2017

@author: Weiyu Lee
"""

import helper
import numpy as np
import problem_unittests as tests

def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    # TODO: Implement Function
    data_max = np.max(x)
    data_min = np.min(x)
    x = (x - data_min) / (data_max - data_min)
    return x

# Unit test function
#tests.test_normalize(normalize)

def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    # TODO: Implement Function
    output = np.zeros((len(x), 10))
    
    for i, j in enumerate(x):
        output[i,j] = 1
           
    return output

#tests.test_one_hot_encode(one_hot_encode)
