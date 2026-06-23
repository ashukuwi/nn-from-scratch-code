"""
Day 7 - What Does "Learning" Even Mean?
From the blog series "Neural Networks, One Day at a Time"
Code from the Day 7 article.

Run this file:  python day_07_what_is_learning.py
"""

target = 0.7      # the answer we want the weight to learn
weight = 0.0      # start with a useless random-ish value
rate   = 0.1      # how big a nudge to make each loop

for attempt in range(11):
    prediction = weight                 # our toy "network" just outputs the weight
    loss = (prediction - target) ** 2   # how wrong we are, as one number
    print(f"attempt {attempt}: weight={weight:.3f}  loss={loss:.4f}")

    # nudge the weight in the direction that lowers the loss
    weight = weight - rate * 2 * (prediction - target)
