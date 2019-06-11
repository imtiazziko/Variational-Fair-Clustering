# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

n_samples = 400
#n_bins = 3  # use 3 bins for calibration_curve as we have 3 clusters here

# Generate 3 blobs with 2 classes where the second blob contains
# half positive samples and half negative samples. Probability in this
# blob is therefore 0.5.
centers = [(1, 1), (3, 1), (1, 5), (3, 5)]
X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=0.2,
                  centers=centers, shuffle=False, random_state=1)

index = n_samples//2
y[0:index] = 0
y[index:n_samples] = 1


colors = ['red','blue']
color_list = np.repeat(colors,len(y)//2)

plt.scatter(X[:,0],X[:,1], c = color_list, label=y)