"""
Day 6 - A Worked Example: A Tiny Network by Hand
From the blog series "Neural Networks, One Day at a Time"

A 2-input, 2-hidden-neuron, 1-output network computed step by step,
then double-checked with numpy matrix math.

Run this file:  python day_06_tiny_network.py
(Requires: numpy  ->  pip install numpy)
"""

import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# --- Inputs ---
x = np.array([0.5, 0.9])

# --- Hidden layer weights/biases (2 inputs -> 2 hidden neurons) ---
W1 = np.array([[0.3, 0.7],    # x1 -> hidden1, hidden2
               [0.4, 0.2]])   # x2 -> hidden1, hidden2
b1 = np.array([0.1, 0.1])

# --- Output layer weights/bias (2 hidden -> 1 output) ---
W2 = np.array([0.6, 0.9])
b2 = 0.2

# ---------- Step by step, by hand ----------
h1_in = x[0] * 0.3 + x[1] * 0.4 + 0.1
h2_in = x[0] * 0.7 + x[1] * 0.2 + 0.1
h1, h2 = sigmoid(h1_in), sigmoid(h2_in)
print("Hidden neuron 1:", round(float(h1), 4))
print("Hidden neuron 2:", round(float(h2), 4))

out_in = h1 * 0.6 + h2 * 0.9 + 0.2
out = sigmoid(out_in)
print("Output:         ", round(float(out), 4))

# ---------- The same thing with numpy matrix math ----------
hidden = sigmoid(x @ W1 + b1)
output = sigmoid(hidden @ W2 + b2)
print("\nNumpy check -> hidden:", np.round(hidden, 4),
      "| output:", round(float(output), 4))
