"""
Day 11 - The Learning Rate: Step Size Matters
From the blog series "Neural Networks, One Day at a Time"
Code from the Day 11 article.

Run this file:  python day_11_learning_rate.py
"""


def descend(lr, steps=7, w=3.0):
    print(f"\nlearning rate = {lr}")
    for s in range(steps):
        slope = 2 * w               # gradient of L = w^2
        w = w - lr * slope          # the update rule
        print(f"  step {s}: w = {w:6.3f},  loss = {w**2:7.3f}")


descend(0.05)   # too small  -> loss barely moves
descend(0.25)   # just right -> loss dives to ~0
descend(1.05)   # too big    -> loss explodes upward
