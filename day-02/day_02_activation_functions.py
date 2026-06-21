"""
Day 2 - Activation Functions: Why Neurons Need a "Decision"
From the blog series "Neural Networks, One Day at a Time"

Run this file:  python day_02_activation_functions.py
(Requires: numpy, matplotlib  ->  pip install numpy matplotlib)
"""

import numpy as np
import matplotlib.pyplot as plt


def step(z):
    """The step function: fire (1) if z reaches 0, otherwise stay quiet (0)."""
    return np.where(z >= 0, 1, 0)


def sigmoid(z):
    """The sigmoid function: a smooth S-curve that eases from 0 up to 1."""
    return 1 / (1 + np.exp(-z))


def neuron(inputs, weights, bias, activation):
    """A single neuron: multiply inputs by weights, add bias, then activate."""
    z = np.dot(inputs, weights) + bias   # the "multiply and add" step
    return activation(z)                 # the "decision" step


if __name__ == "__main__":
    # A neuron with two inputs
    inputs = np.array([1.0, 2.0])
    weights = np.array([0.5, -0.3])
    bias = 0.1

    z = np.dot(inputs, weights) + bias
    print("Raw score z   =", round(float(z), 3))
    print("With step     =", neuron(inputs, weights, bias, step))
    print("With sigmoid  =", round(float(neuron(inputs, weights, bias, sigmoid)), 3))

    # Plot both activation functions so you can SEE the difference
    grid = np.linspace(-6, 6, 200)
    plt.figure(figsize=(8, 4))
    plt.plot(grid, step(grid), label="step")
    plt.plot(grid, sigmoid(grid), label="sigmoid")
    plt.title("Step vs. Sigmoid")
    plt.xlabel("z (raw score)")
    plt.ylabel("output")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
