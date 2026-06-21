"""
Day 3 - The Usual Suspects: Sigmoid, Tanh, and ReLU
From the blog series "Neural Networks, One Day at a Time"

Run this file:  python day_03_sigmoid_tanh_relu.py
(Requires: numpy, matplotlib  ->  pip install numpy matplotlib)
"""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    """Outputs between 0 and 1. Great for probabilities."""
    return 1 / (1 + np.exp(-z))


def tanh(z):
    """Outputs between -1 and 1. Centered on zero."""
    return np.tanh(z)


def relu(z):
    """Outputs 0 for negatives, otherwise passes the value straight through."""
    return np.maximum(0, z)


if __name__ == "__main__":
    # Compare what each function does to the same few inputs
    for value in [-2.0, 0.0, 2.0]:
        print(
            f"z={value:+.0f} | "
            f"sigmoid={sigmoid(value):.3f} | "
            f"tanh={tanh(value):+.3f} | "
            f"relu={relu(value):.3f}"
        )

    # Plot all three together
    grid = np.linspace(-6, 6, 200)
    plt.figure(figsize=(8, 4))
    plt.plot(grid, sigmoid(grid), label="sigmoid (0 to 1)")
    plt.plot(grid, tanh(grid), label="tanh (-1 to 1)")
    plt.plot(grid, relu(grid), label="ReLU (0 and up)")
    plt.title("Three Common Activation Functions")
    plt.xlabel("z")
    plt.ylabel("output")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
