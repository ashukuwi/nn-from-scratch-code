"""
Day 13 - Backpropagation, Part 2: Following the Chain
From the blog series "Neural Networks, One Day at a Time"
Code from the Day 13 article.

Run this file:  python day_13_chain_rule.py
"""

# A tiny chain:  w -> z -> a -> L
# Each local slope: how much the next step moves when this one nudges up.
dz_dw = 2.0   # the sum z reacts to the weight w
da_dz = 0.5   # the activation a reacts to the sum z
dL_da = 0.4   # the loss L reacts to the activation a

# Backward pass: start at the loss, multiply as we travel back.
grad = 1.0           # the loss compared to itself
grad = grad * dL_da  # blame on a   -> 0.4
grad = grad * da_dz  # blame on z   -> 0.2
grad = grad * dz_dw  # slope for w  -> 0.4

print("Slope (gradient) for w:", grad)   # 0.4
