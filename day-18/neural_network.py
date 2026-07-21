"""
Neural Network — cumulative program (through Day 18)
From the blog series "Neural Networks, One Day at a Time"

The ONE running program the series grows over time.

  Through Day 15: a 2 -> 8 -> 1 network that learns XOR with a full training loop.
  Day 16: split the data into TRAIN / VALIDATION / TEST before training.
  Day 17: diagnose underfitting vs overfitting from the train/validation gap.
  NEW on Day 18: add L2 regularization (weight decay) so the network is taxed
    for oversized weights and stays humble — the penalty gradient 2*lam*W is
    added to each weight's gradient during the update.

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

# ---- Day 16: split 70% train / 15% validation / 15% test ----
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
lam = 0.0003                                         # NEW (Day 18): L2 strength (mild)


def forward(inputs):
    h = sigmoid(inputs @ W1 + b1)
    return h, sigmoid(h @ W2 + b2)


def loss_on(inputs, targets):                        # mean squared error
    _, pred = forward(inputs)
    return np.mean((pred - targets) ** 2)


def accuracy(inputs, targets):                       # fraction predicted correctly
    _, pred = forward(inputs)
    return np.mean((pred > 0.5) == (targets > 0.5))


# ---- The training loop (trains ONLY on the training set) ----
for epoch in range(n_epochs):
    order = np.random.permutation(len(X_train))
    Xs, ys = X_train[order], y_train[order]

    for start in range(0, len(Xs), batch_size):
        xb = Xs[start:start + batch_size]
        yb = ys[start:start + batch_size]

        h    = sigmoid(xb @ W1 + b1)                 # 1. forward pass
        pred = sigmoid(h @ W2 + b2)
        d_out = 2 * (pred - yb) / len(xb) * pred * (1 - pred)   # 3. backprop
        dW2 = h.T @ d_out;   db2 = d_out.sum(axis=0, keepdims=True)
        d_h = (d_out @ W2.T) * h * (1 - h)
        dW1 = xb.T @ d_h;    db1 = d_h.sum(axis=0, keepdims=True)

        dW2 += 2 * lam * W2                          # NEW (Day 18): L2 penalty gradient
        dW1 += 2 * lam * W1                          # (biases are left unregularized)

        W2 -= learning_rate * dW2;  b2 -= learning_rate * db2   # 4. update
        W1 -= learning_rate * dW1;  b1 -= learning_rate * db1

    if epoch % 50 == 0 or epoch == n_epochs - 1:
        print(f"Epoch {epoch:3d}:  train loss = {loss_on(X_train, y_train):.4f}"
              f"   val loss = {loss_on(X_val, y_val):.4f}")

# ---- Day 17: diagnose the fit from the train/validation gap ----
train_acc = accuracy(X_train, y_train)
val_acc   = accuracy(X_val, y_val)
gap = train_acc - val_acc
print(f"\nTrain accuracy = {train_acc * 100:.1f}%   Validation accuracy = {val_acc * 100:.1f}%")
if train_acc < 0.75:
    print("Diagnosis: UNDERFITTING — poor even on training data -> grow/train more")
elif gap > 0.10:
    print("Diagnosis: OVERFITTING — great on training, weak on validation -> regularize")
else:
    print("Diagnosis: GOOD BALANCE — train and validation scores are close and high")

# ---- Day 18: L2 keeps weights modest — show the average weight size ----
mean_weight = (np.abs(W1).mean() + np.abs(W2).mean()) / 2
print(f"Average |weight| = {mean_weight:.3f}  (L2 weight decay keeps this small)")

# ---- Day 16: open the TEST set exactly ONCE, at the very end ----
print(f"\nFinal test accuracy = {accuracy(X_test, y_test) * 100:.1f}%  (data the network never trained on)")
