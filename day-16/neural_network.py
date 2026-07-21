"""
Neural Network — cumulative program (through Day 16)
From the blog series "Neural Networks, One Day at a Time"

This is the ONE running program the series grows over time. Each day it gains
the new capability from that day's article, building in both length and rigor.

  Through Day 15: a 2 -> 8 -> 1 network that learns the XOR pattern with a full
    training loop (forward pass, loss, backprop, weight update, epochs & batches).
  NEW on Day 16: the data is split into TRAIN / VALIDATION / TEST sets BEFORE
    training, so we can measure honest generalization instead of memorization.
    We train only on the training set, watch the validation loss as we go, and
    open the test set exactly once at the very end for an unbiased score.

Run:  python neural_network.py      (requires: numpy)
"""

import numpy as np

# ---- A dataset that NEEDS a hidden layer: noisy XOR ----
np.random.seed(0)
corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels  = np.array([[0], [1], [1], [0]])            # the XOR pattern
idx = np.random.randint(0, 4, size=500)
X = corners[idx] + 0.1 * np.random.randn(500, 2)    # 500 noisy points
y = labels[idx]

# ---- NEW (Day 16): split 70% train / 15% validation / 15% test ----
perm = np.random.permutation(len(X))
X, y = X[perm], y[perm]
n_train = int(0.70 * len(X))                         # 350
n_val   = int(0.15 * len(X))                         # 75
X_train, y_train = X[:n_train],               y[:n_train]
X_val,   y_val   = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
X_test,  y_test  = X[n_train + n_val:],        y[n_train + n_val:]
print(f"Split: {len(X_train)} train, {len(X_val)} validation, {len(X_test)} test")


# ---- The model: 2 inputs -> 8 hidden neurons -> 1 output ----
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


W1 = np.random.randn(2, 8) * 0.5;  b1 = np.zeros((1, 8))
W2 = np.random.randn(8, 1) * 0.5;  b2 = np.zeros((1, 1))
learning_rate = 1.0
batch_size = 40
n_epochs = 300


def forward(inputs):
    h = sigmoid(inputs @ W1 + b1)
    return h, sigmoid(h @ W2 + b2)


def loss_on(inputs, targets):                        # mean squared error
    _, pred = forward(inputs)
    return np.mean((pred - targets) ** 2)


# ---- The training loop (trains ONLY on the training set) ----
for epoch in range(n_epochs):
    order = np.random.permutation(len(X_train))      # reshuffle each epoch
    Xs, ys = X_train[order], y_train[order]

    for start in range(0, len(Xs), batch_size):      # inner loop over batches
        xb = Xs[start:start + batch_size]
        yb = ys[start:start + batch_size]

        h    = sigmoid(xb @ W1 + b1)                 # 1. forward pass
        pred = sigmoid(h @ W2 + b2)
        d_out = 2 * (pred - yb) / len(xb) * pred * (1 - pred)   # 3. backprop
        dW2 = h.T @ d_out;   db2 = d_out.sum(axis=0, keepdims=True)
        d_h = (d_out @ W2.T) * h * (1 - h)
        dW1 = xb.T @ d_h;    db1 = d_h.sum(axis=0, keepdims=True)
        W2 -= learning_rate * dW2;  b2 -= learning_rate * db2   # 4. update
        W1 -= learning_rate * dW1;  b1 -= learning_rate * db1

    if epoch % 50 == 0 or epoch == n_epochs - 1:
        # NEW (Day 16): watch train vs validation loss side by side
        print(f"Epoch {epoch:3d}:  train loss = {loss_on(X_train, y_train):.4f}"
              f"   val loss = {loss_on(X_val, y_val):.4f}")

# ---- NEW (Day 16): open the TEST set exactly ONCE, at the very end ----
_, test_pred = forward(X_test)
test_acc = np.mean((test_pred > 0.5) == (y_test > 0.5))
print(f"\nFinal test loss     = {loss_on(X_test, y_test):.4f}")
print(f"Final test accuracy = {test_acc * 100:.1f}%  (on data the network never trained on)")
