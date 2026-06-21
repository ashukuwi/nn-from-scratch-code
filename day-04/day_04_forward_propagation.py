"""
Day 4 - Forward Propagation: Pushing Data Through the Network
From the blog series "Neural Networks, One Day at a Time"

Walk one example from the input layer to the output, layer by layer.

Run this file:  python day_04_forward_propagation.py
(Requires: numpy  ->  pip install numpy)
"""

import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward(inputs):
    # --- Layer 1: 2 inputs -> 3 hidden neurons ---
    W1 = np.array([[0.2, 0.4, -0.5],
                   [0.6, -0.3, 0.8]])   # shape (2 inputs, 3 neurons)
    b1 = np.array([0.1, -0.2, 0.05])
    z1 = inputs @ W1 + b1                # multiply and add for all 3 at once
    a1 = sigmoid(z1)                     # activate the hidden layer

    # --- Layer 2: 3 hidden neurons -> 1 output ---
    W2 = np.array([[0.7],
                   [-0.6],
                   [0.3]])               # shape (3 neurons, 1 output)
    b2 = np.array([0.2])
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)                     # final prediction
    return a1, a2


if __name__ == "__main__":
    inputs = np.array([1.0, 0.5])
    hidden, output = forward(inputs)
    print("Input:         ", inputs)
    print("Hidden layer:  ", np.round(hidden, 3))
    print("Output (pred): ", np.round(output, 3))
