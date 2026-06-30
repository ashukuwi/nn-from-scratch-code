"""
Day 15 - Putting It Together: One Full Training Loop
From the blog series "Neural Networks, One Day at a Time"
Code from the Day 15 article.

Run this file:  python day_15_training_loop.py
(Requires: numpy  ->  pip install numpy)
"""

import numpy as np

# ---- A dataset that NEEDS a hidden layer: noisy XOR ----
np.random.seed(0)
corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels  = np.array([[0], [1], [1], [0]])           # the XOR pattern
idx = np.random.randint(0, 4, size=400)
X = corners[idx] + 0.1 * np.random.randn(400, 2)   # 400 noisy points
y = labels[idx]


# ---- The model: 2 inputs -> 8 hidden neurons -> 1 output ----
def sigmoid(z): return 1 / (1 + np.exp(-z))


W1 = np.random.randn(2, 8) * 0.5;  b1 = np.zeros((1, 8))
W2 = np.random.randn(8, 1) * 0.5;  b2 = np.zeros((1, 1))
learning_rate = 1.0
batch_size = 40
n_epochs = 300

for epoch in range(n_epochs):                      # repeat the whole pass many times
    order = np.random.permutation(len(X))          # reshuffle so batches vary each pass
    Xs, ys = X[order], y[order]

    for start in range(0, len(Xs), batch_size):    # inner loop over batches
        xb = Xs[start:start + batch_size]
        yb = ys[start:start + batch_size]

        # 1. FORWARD PASS — push the batch through both layers
        h    = sigmoid(xb @ W1 + b1)               # hidden layer
        pred = sigmoid(h @ W2 + b2)                # output layer

        # 2. COMPUTE LOSS — one number: how wrong were we?
        loss = np.mean((pred - yb) ** 2)

        # 3. BACKPROPAGATION — carry each weight's share of the blame backward
        d_out = 2 * (pred - yb) / len(xb) * pred * (1 - pred)
        dW2 = h.T @ d_out;   db2 = d_out.sum(axis=0, keepdims=True)
        d_h = (d_out @ W2.T) * h * (1 - h)         # the chain rule, one layer back
        dW1 = xb.T @ d_h;    db1 = d_h.sum(axis=0, keepdims=True)

        # 4. UPDATE WEIGHTS — nudge every weight a small step downhill
        W2 -= learning_rate * dW2;  b2 -= learning_rate * db2
        W1 -= learning_rate * dW1;  b1 -= learning_rate * db1

    if epoch % 50 == 0 or epoch == n_epochs - 1:
        print(f"Epoch {epoch:3d}: loss = {loss:.4f}")   # watch it fall

# ---- Did it learn? Check the four clean XOR cases ----
h = sigmoid(corners @ W1 + b1)
print("\nPredictions:", sigmoid(h @ W2 + b2).round(2).ravel(), " (want 0, 1, 1, 0)")
