"""
Day 5 - Representing It All as Math: Vectors and Matrices
From the blog series "Neural Networks, One Day at a Time"

Shows that ONE matrix multiply replaces many little multiply-and-add steps.

Run this file:  python day_05_vectors_and_matrices.py
(Requires: numpy  ->  pip install numpy)
"""

import numpy as np

inputs = np.array([1.0, 2.0, 3.0])

# 3 inputs feeding 2 neurons -> weight matrix of shape (3, 2)
W = np.array([[0.1, 0.4],
              [0.2, 0.5],
              [0.3, 0.6]])
b = np.array([0.5, -0.5])

# The slow way: loop over every neuron and every input by hand
manual = []
for j in range(W.shape[1]):          # for each neuron
    total = b[j]
    for i in range(len(inputs)):     # for each input
        total += inputs[i] * W[i, j]
    manual.append(total)

# The fast way: a single matrix multiply does all of that at once
fast = inputs @ W + b

print("Manual loop result:", np.round(manual, 3))
print("Matrix multiply   :", np.round(fast, 3))
print("Same answer?       ", np.allclose(manual, fast))
