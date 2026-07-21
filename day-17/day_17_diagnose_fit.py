"""
Day 17 - Overfitting and Underfitting: Diagnosing the Train/Validation Gap
From the blog series "Neural Networks, One Day at a Time"
Standalone, runnable version of the Day 17 diagnostic.

The article showed this logic with placeholder evaluate() calls; here it runs
as-is with real example scores. Swap in your own model's train/val accuracy.

Run this file:  python day_17_diagnose_fit.py
"""

# After training, compare performance on the two sets.
train_acc = 0.98   # accuracy on data the model trained on
val_acc   = 0.74   # accuracy on held-out validation data

gap = train_acc - val_acc

if train_acc < 0.75:
    print("Underfitting: poor even on training data -> grow/train more")
elif gap > 0.10:
    print("Overfitting: great on training, weak on validation -> regularize")
else:
    print("Good balance: train and validation scores are close and high")
