"""
Day 18 - Regularization: Adding an L2 Penalty to the Loss
From the blog series "Neural Networks, One Day at a Time"
Standalone, runnable version of the Day 18 snippet.

The total_loss function is exactly as shown in the article; the example at the
bottom makes it runnable so you can see big weights cost more.

Run this file:  python day_18_regularization.py
(Requires: numpy)
"""

import numpy as np

lam = 0.01  # lambda: the regularization strength (the "tax rate")


def total_loss(predictions, targets, weights):
    # 1) the ordinary error, e.g. mean squared error from Day 8
    data_loss = np.mean((predictions - targets) ** 2)

    # 2) the L2 penalty: sum of every weight squared
    l2_penalty = lam * np.sum(weights ** 2)
    # (for L1, you would use: lam * np.sum(np.abs(weights)))

    return data_loss + l2_penalty


# --- A concrete, runnable example: same error, different weight sizes ---
predictions = np.array([0.9, 0.2, 0.3])
targets     = np.array([1.0, 0.0, 1.0])
small_weights = np.array([0.3, -0.2, 0.5, 0.1])
big_weights   = np.array([3.0, -2.5, 4.0, 1.5])

data_only = np.mean((predictions - targets) ** 2)
print(f"data loss only            = {data_only:.4f}")
print(f"total loss, small weights = {total_loss(predictions, targets, small_weights):.4f}")
print(f"total loss, big weights   = {total_loss(predictions, targets, big_weights):.4f}")
print("Same predictions, but bigger weights pay a bigger penalty.")
