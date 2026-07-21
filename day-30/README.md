# Day 30 — The Complete Neural Network (Capstone)

This folder holds the finished capstone from Day 30 of the series: a neural network built completely from scratch (NumPy only) that classifies the **Wine dataset** into its three grape cultivars. It ties together everything from the series: scaling, a train/validation/test split, weights and biases, ReLU and softmax, the forward pass, cross-entropy loss, backpropagation, gradient descent, mini-batches, L2 regularization, and early stopping.

There are two copies of the exact same program:

- **`day_30_full_network.py`** — a plain Python script.
- **`day_30_full_network.ipynb`** — the same code as a notebook, ideal for Google Colab.

## What you need

Python 3 with two libraries:

- **NumPy** — does all the maths.
- **scikit-learn** — used for one thing only, loading the Wine dataset.

Install them with:

```bash
pip install numpy scikit-learn
```

## Option 1: Run the `.py` file on your computer

1. Open a terminal in this `day-30` folder.
2. Run:

   ```bash
   python day_30_full_network.py
   ```

3. You will see the training progress print out epoch by epoch, then the final test accuracy and a confusion matrix.

## Option 2: Run the `.ipynb` file in your browser (no installing)

1. Go to https://colab.research.google.com
2. Choose **File → Upload notebook** and pick `day_30_full_network.ipynb` (or click the "Open in Colab" badge in the main repo README).
3. Press the run button on the cell, or use **Runtime → Run all**. Colab already has NumPy and scikit-learn installed, so there is nothing to set up.

## Try changing things

Once it runs, experiment: make the hidden layer bigger or smaller, raise or lower the learning rate, or comment out the scaling step and watch training struggle. Every knob is a concept from an earlier day in the series.
