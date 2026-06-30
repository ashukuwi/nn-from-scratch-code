"""
Day 9 - Cross-Entropy and Classification Loss
From the blog series "Neural Networks, One Day at a Time"
Code from the Day 9 article.

Run this file:  python day_09_cross_entropy.py
"""

import math


def cross_entropy(probs, true_index):
    p = probs[true_index]          # confidence given to the CORRECT class
    return -math.log(p)            # the whole loss, in one line


# Network is fairly right: 0.70 on the true class (Cat = index 0)
print(round(cross_entropy([0.70, 0.20, 0.10], 0), 3))   # -> 0.357

# Network is confidently WRONG: only 0.05 on the true class
print(round(cross_entropy([0.05, 0.90, 0.05], 0), 3))   # -> 2.996
