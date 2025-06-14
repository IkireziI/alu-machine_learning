#! /usr/bin/env python3

import numpy as np

def variance(X, C, clss=None):
    """
    calculates intra-cluster variance for a dataset
    
    X: numpy.ndarray (n, d) - data points
    C: numpy.ndarray (k, d) - centroids
    clss: numpy.ndarray (n,) - index of the nearest centroid for each point
    
    return:
        var (float) - total intra-cluster variance
    """
    if clss is None:
        raise ValueError("clustering IDs (clss) must be provided with centroid positions.")
  
    var = 0.0
    for k in range(len(C)):
        var += np.sum((X[clss == k] - C[k]) ** 2)
    return var
