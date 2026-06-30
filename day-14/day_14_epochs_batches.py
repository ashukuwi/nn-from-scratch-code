"""
Day 14 - Epochs, Batches, and Iterations
From the blog series "Neural Networks, One Day at a Time"
Code from the Day 14 article.

Run this file:  python day_14_epochs_batches.py
(Requires: numpy  ->  pip install numpy)
"""

import numpy as np

# ---- A tiny dataset: 1,000 examples from the line y = 2x + 1 (plus noise) ----
np.random.seed(0)
X = np.random.uniform(-1, 1, size=(1000, 1))      # 1,000 inputs
y = 2 * X + 1 + 0.1 * np.random.randn(1000, 1)    # the "true" answers

# ---- The model: a single weight and bias we will learn ----
w, b = 0.0, 0.0
learning_rate = 0.1
batch_size = 100
n_epochs = 5


def forward_pass(x):                 # make a guess
    return w * x + b


def compute_loss(pred, target):      # measure how wrong (mean squared error)
    return np.mean((pred - target) ** 2)


def get_batches(X, y, size):         # chop the data into mini-batches
    for start in range(0, len(X), size):
        yield X[start:start + size], y[start:start + size]


iteration = 0
for epoch in range(n_epochs):                      # one epoch = one full pass over the data
    order = np.random.permutation(len(X))          # reshuffle so batches differ each epoch
    X, y = X[order], y[order]

    for xb, yb in get_batches(X, y, batch_size):   # inner loop over batches
        pred   = forward_pass(xb)                  # 1) make a guess
        loss   = compute_loss(pred, yb)            # 2) measure how wrong
        grad_w = np.mean(2 * (pred - yb) * xb)     # 3) slope of the loss for w
        grad_b = np.mean(2 * (pred - yb))          #    slope of the loss for b
        w -= learning_rate * grad_w                # 4) one weight update
        b -= learning_rate * grad_b
        iteration += 1                             # reaching here once = ONE ITERATION
    # finishing the inner loop once = ONE EPOCH
    print(f"Epoch {epoch+1}: loss = {loss:.4f}   (iterations so far: {iteration})")

print(f"\nDone. {n_epochs} epochs x 10 batches = {iteration} total weight updates.")
print(f"Learned w = {w:.2f}, b = {b:.2f}   (true values were 2 and 1)")
