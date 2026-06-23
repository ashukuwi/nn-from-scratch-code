"""
Day 8 - Loss Functions: Measuring How Wrong We Are
From the blog series "Neural Networks, One Day at a Time"
Code from the Day 8 article.

Run this file:  python day_08_loss_functions.py
"""

predictions = [0.9, 0.2, 0.3]   # what the network guessed
actuals     = [1.0, 0.0, 1.0]   # the correct answers

squared_errors = []
for p, a in zip(predictions, actuals):
    error = p - a               # 1. subtract to get the gap
    squared_errors.append(error ** 2)   # 2. square it

mse = sum(squared_errors) / len(squared_errors)   # 3. take the mean
print(f"loss (MSE) = {mse:.2f}")   # -> loss (MSE) = 0.18
