"""
Day 10 - Gradient Descent: Rolling Downhill to Better Weights
From the blog series "Neural Networks, One Day at a Time"
Code from the Day 10 article.

Run this file:  python day_10_gradient_descent.py
"""

w = 3.0            # starting weight, up on the hill
learning_rate = 0.22

for step in range(6):
    slope = 2 * w                      # the gradient at this spot
    w = w - learning_rate * slope      # step OPPOSITE the slope
    print(f"step {step}: w = {w:.3f},  loss = {w**2:.3f}")

# step 0: w = 1.680, loss = 2.822
# step 1: w = 0.941, loss = 0.885
# step 2: w = 0.527, loss = 0.278
# step 3: w = 0.295, loss = 0.087
# step 4: w = 0.165, loss = 0.027
# step 5: w = 0.093, loss = 0.009
