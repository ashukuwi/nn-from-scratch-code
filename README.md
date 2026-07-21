# Neural Networks From Scratch — Code

Runnable Python code from my daily blog series **"Neural Networks, One Day at a Time"**, where I break neural networks down for beginners. Built from scratch as I teach myself AI in high school.

Each folder maps to one article in the series. Every day has two ways to run it:

- a plain **`.py`** file you can run on your own machine, and
- a **Google Colab notebook** (`.ipynb`) you can run directly in your browser — no setup required.

## Days

| Day | Topic | Run in Colab |
|----|-------|--------------|
| 02 | Activation Functions: step & sigmoid | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashukuwi/nn-from-scratch-code/blob/main/day-02/day_02_activation_functions.ipynb) |
| 03 | Sigmoid, Tanh, and ReLU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashukuwi/nn-from-scratch-code/blob/main/day-03/day_03_sigmoid_tanh_relu.ipynb) |
| 04 | Forward Propagation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashukuwi/nn-from-scratch-code/blob/main/day-04/day_04_forward_propagation.ipynb) |
| 05 | Vectors and Matrices | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashukuwi/nn-from-scratch-code/blob/main/day-05/day_05_vectors_and_matrices.ipynb) |
| 06 | A Tiny Network by Hand | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashukuwi/nn-from-scratch-code/blob/main/day-06/day_06_tiny_network.ipynb) |
| 07 | What Does "Learning" Even Mean? | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashukuwi/nn-from-scratch-code/blob/main/day-07/day_07_what_is_learning.ipynb) |
| 08 | Loss Functions (MSE) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashukuwi/nn-from-scratch-code/blob/main/day-08/day_08_loss_functions.ipynb) |
| 09 | Cross-Entropy and Classification Loss | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashukuwi/nn-from-scratch-code/blob/main/day-09/day_09_cross_entropy.ipynb) |
| 10 | Gradient Descent | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashukuwi/nn-from-scratch-code/blob/main/day-10/day_10_gradient_descent.ipynb) |
| 11 | The Learning Rate | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashukuwi/nn-from-scratch-code/blob/main/day-11/day_11_learning_rate.ipynb) |
| 13 | Backpropagation: The Chain Rule | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashukuwi/nn-from-scratch-code/blob/main/day-13/day_13_chain_rule.ipynb) |
| 14 | Epochs, Batches, and Iterations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashukuwi/nn-from-scratch-code/blob/main/day-14/day_14_epochs_batches.ipynb) |
| 15 | One Full Training Loop | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashukuwi/nn-from-scratch-code/blob/main/day-15/day_15_training_loop.ipynb) |
| 16 | Train / Validation / Test Split — builds the *cumulative network* | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashukuwi/nn-from-scratch-code/blob/main/day-16/neural_network.ipynb) |
| 17 | Overfitting & Underfitting — [snippet](https://colab.research.google.com/github/ashukuwi/nn-from-scratch-code/blob/main/day-17/day_17_diagnose_fit.ipynb) · [cumulative net](https://colab.research.google.com/github/ashukuwi/nn-from-scratch-code/blob/main/day-17/neural_network.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashukuwi/nn-from-scratch-code/blob/main/day-17/neural_network.ipynb) |
| 18 | Regularization (L2) — [snippet](https://colab.research.google.com/github/ashukuwi/nn-from-scratch-code/blob/main/day-18/day_18_regularization.ipynb) · [cumulative net](https://colab.research.google.com/github/ashukuwi/nn-from-scratch-code/blob/main/day-18/neural_network.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashukuwi/nn-from-scratch-code/blob/main/day-18/neural_network.ipynb) |
| 30 | The Complete Network — capstone on the Wine dataset ([run instructions](https://github.com/ashukuwi/nn-from-scratch-code/blob/main/day-30/README.md)) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashukuwi/nn-from-scratch-code/blob/main/day-30/day_30_full_network.ipynb) |

## The cumulative program

Starting with Day 16, each day's folder also contains **`neural_network.py`** — one continually growing program that carries everything the series has built so far and adds that day's new capability, so it grows in both length and rigor over time. Day 15's full training loop is its starting point; Day 16 adds splitting the data into train/validation/test sets before training, so the network is measured on data it never learned from. Each day, this file will pick up the next component (regularization, dropout, and so on).

*(Day 12 and Day 16 have no standalone code snippet in their articles. Day 12 is intuition-only; Day 16's contribution lives in the cumulative program above.)*

More days are added as the series is published.

## Running locally

You'll need Python 3 with NumPy and Matplotlib:

```bash
pip install numpy matplotlib
python day-02/day_02_activation_functions.py
```

## Running in your browser

Click any **Open in Colab** badge above. Colab already has NumPy and Matplotlib installed, so you can just press **Run** — nothing to set up.

## About

I'm a high schooler documenting my journey learning AI and neural networks. The goal of this repo is to keep every example simple, readable, and easy to run so other beginners can follow along and experiment.
