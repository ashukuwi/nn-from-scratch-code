# =====================================================================
# Day 30 capstone: a complete neural network, built from scratch.
# 13 wine measurements -> 16 hidden neurons (ReLU) -> 3 cultivars (softmax)
# The only library we need for the maths is NumPy. scikit-learn is used
# for one thing only: handing us the Wine dataset.
# =====================================================================

import numpy as np                          # NumPy: fast maths on whole arrays of numbers at once
from sklearn.datasets import load_wine      # the ready-made Wine dataset (178 wines, 13 measurements each)

# A "seed" fixes the random number generator, so the shuffling and the random
# starting weights come out the same every time you run this file.
# Without it you would get slightly different numbers on every run.
np.random.seed(42)


# ---------------------------------------------------------------------
# LOAD THE DATA AND TAKE A FIRST LOOK
# ---------------------------------------------------------------------

wine = load_wine()          # loads the dataset into a bundle object
X = wine.data               # X = the inputs: a table with one row per wine and 13 columns of measurements
y = wine.target             # y = the answers: one number per wine, 0, 1 or 2, saying which cultivar it is

# .shape tells us the size of a table as (rows, columns).
print("Inputs  X shape:", X.shape)          # expect (178, 13): 178 wines, 13 measurements each
print("Answers y shape:", y.shape)          # expect (178,): one answer per wine
print("Feature names:", list(wine.feature_names))   # the human-readable name of each of the 13 columns
print("Class names:", list(wine.target_names))      # the names of the 3 cultivars

# np.bincount counts how many times each whole number appears in a list.
# Here it tells us how many wines belong to each of the three cultivars.
print("Wines per cultivar:", np.bincount(y))

# Peek at the very first wine so the numbers stop being abstract.
# X[0] means "row number 0" (Python counts from 0, so this is the first wine).
print("First wine's 13 measurements:", X[0])
print("First wine's true cultivar:", y[0])


# ---------------------------------------------------------------------
# STEP 2. SHUFFLE AND SPLIT INTO TRAIN / VALIDATION / TEST
#    (this runs before the scaling, because the averages below are only
#     allowed to look at the training wines)
# ---------------------------------------------------------------------

n_samples = X.shape[0]                       # how many wines we have in total (178)

# np.random.permutation makes a shuffled list of the row numbers 0..177.
# The wines are stored grouped by cultivar, so shuffling is essential:
# otherwise the test set would be one single cultivar.
shuffled_indices = np.random.permutation(n_samples)

X = X[shuffled_indices]                      # reorder the inputs using that shuffled order
y = y[shuffled_indices]                      # reorder the answers exactly the same way, so they still line up

# Work out where to cut the shuffled pile: 70% train, 15% validation, 15% test.
# int(...) throws away the decimal part, because we need whole numbers of wines.
n_train = int(0.70 * n_samples)              # 124 wines to learn from
n_val = int(0.15 * n_samples)                # 26 wines to tune with

# Slicing with [a:b] takes the rows from position a up to (but not including) b.
X_train, y_train = X[:n_train], y[:n_train]                          # the first 70%
X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]  # the next 15%
X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]              # everything left over

print("Train / validation / test sizes:", len(X_train), len(X_val), len(X_test))


# ---------------------------------------------------------------------
# STEP 1. STANDARDIZE THE FEATURES (measure on the training data only)
# ---------------------------------------------------------------------

# .mean(axis=0) averages each column separately, giving 13 averages (one per feature).
# "axis=0" means "go down the rows"; axis=1 would mean "go across the columns".
mean = X_train.mean(axis=0)

# .std(axis=0) is the standard deviation of each column: how spread out that feature is.
std = X_train.std(axis=0)

# Subtract the average and divide by the spread. Every feature now sits around 0
# with a similar range, so no feature bullies the others just because of its units.
# NumPy applies this row by row automatically ("broadcasting").
X_train = (X_train - mean) / std

# CRUCIAL: the validation and test sets are rescaled with the TRAINING mean and std,
# never their own. Using their own would leak information about data we are
# supposed to be pretending we have never seen.
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std


# ---------------------------------------------------------------------
# STEP 3. BUILD THE NETWORK: weights, biases, ReLU and softmax
# ---------------------------------------------------------------------

n_inputs = 13        # 13 measurements come in
n_hidden = 16        # 16 neurons in the middle layer
n_outputs = 3        # 3 cultivars come out

# np.random.randn(a, b) makes an a-by-b table of random numbers centred on 0.
# We shrink them by sqrt(2 / number_of_inputs). This is "He initialization":
# it keeps the signal from exploding or fading as it passes through ReLU layers.
W1 = np.random.randn(n_inputs, n_hidden) * np.sqrt(2.0 / n_inputs)   # weights from inputs -> hidden
b1 = np.zeros(n_hidden)                                             # one bias per hidden neuron, all starting at 0
W2 = np.random.randn(n_hidden, n_outputs) * np.sqrt(2.0 / n_hidden)  # weights from hidden -> outputs
b2 = np.zeros(n_outputs)                                            # one bias per output neuron


def relu(z):
    """ReLU: keep positive numbers as they are, turn negative numbers into 0.
    np.maximum compares each number with 0 and keeps the larger one.
    That little bend is what lets the network learn curved patterns."""
    return np.maximum(0, z)


def softmax(z):
    """Turn each row of raw scores into probabilities that add up to 1.

    z has one row per wine and one column per cultivar."""
    # Subtracting the biggest score in each row changes nothing mathematically
    # but stops np.exp from overflowing on large numbers. keepdims=True keeps the
    # result shaped as a column so it lines up with z row by row.
    z = z - z.max(axis=1, keepdims=True)
    exp_scores = np.exp(z)                                  # np.exp raises e to the power of each number; makes everything positive
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)  # divide each row by its own total, so the row sums to 1


def one_hot(labels, n_classes=3):
    """Turn answers like 2 into a row of 0s with a single 1: [0, 0, 1].
    That shape matches the network's three probability outputs, so we can
    compare them directly."""
    encoded = np.zeros((len(labels), n_classes))   # start with a table of all zeros
    encoded[np.arange(len(labels)), labels] = 1    # in row i, put a 1 in the column named by labels[i]
    return encoded


# Convert the answers once, up front, into that one-hot form.
Y_train = one_hot(y_train)
Y_val = one_hot(y_val)
Y_test = one_hot(y_test)


# ---------------------------------------------------------------------
# STEP 4. THE FORWARD PASS AND THE CROSS-ENTROPY LOSS
# ---------------------------------------------------------------------

def forward(X, W1, b1, W2, b2):
    """Run wines through the network and return its probability guesses.

    We also return the in-between values, because backpropagation needs them."""
    # np.dot is matrix multiplication: every wine's 13 numbers get multiplied by
    # the weights and summed, producing 16 numbers per wine. Then we add the bias.
    z1 = np.dot(X, W1) + b1     # raw scores of the hidden layer (before the activation)
    a1 = relu(z1)               # the hidden layer's actual output, after ReLU
    z2 = np.dot(a1, W2) + b2    # raw scores of the 3 output neurons
    probs = softmax(z2)         # turned into 3 probabilities per wine
    return z1, a1, probs


def cross_entropy_loss(probs, Y_true):
    """How wrong the guesses were, as a single number. Lower is better.

    It looks only at the probability given to the CORRECT cultivar and punishes
    small values harshly: -log(0.9) is tiny, -log(0.01) is huge."""
    # 1e-9 is a whisker added so we never take the log of exactly 0 (which is undefined).
    correct_probs = np.sum(probs * Y_true, axis=1)      # multiplying by the one-hot row keeps only the correct class's probability
    return -np.mean(np.log(correct_probs + 1e-9))       # np.mean averages over the wines, so the loss is per-wine and batch-size independent


# ---------------------------------------------------------------------
# STEP 5. TRAIN IT: mini-batches, backpropagation, L2, early stopping
# ---------------------------------------------------------------------

learning_rate = 0.1       # how big a step we take downhill on each update
batch_size = 16           # how many wines we look at before each update
epochs = 200              # how many full passes over the training data
lam = 1e-3                # lambda: the strength of the L2 penalty on oversized weights

train_losses = []         # we record the loss each epoch so we can plot it later
val_losses = []

best_val_loss = float("inf")   # "infinity": any real loss will beat it on the first epoch
best_weights = None            # here we will stash a copy of the best network we have seen
best_epoch = 0

for epoch in range(epochs):                    # one trip round this loop = one epoch
    # Reshuffle the training wines each epoch so the batches differ every time.
    order = np.random.permutation(len(X_train))
    X_shuffled = X_train[order]
    Y_shuffled = Y_train[order]

    # Walk through the training data in chunks of batch_size: 0, 16, 32, ...
    for start in range(0, len(X_shuffled), batch_size):
        X_batch = X_shuffled[start:start + batch_size]   # this batch's wines
        Y_batch = Y_shuffled[start:start + batch_size]   # this batch's correct answers
        m = len(X_batch)                                 # how many wines are actually in this batch

        # --- forward: make predictions for this batch ---
        z1, a1, probs = forward(X_batch, W1, b1, W2, b2)

        # --- backward: work out how each weight contributed to the error ---
        # For softmax + cross-entropy the maths collapses to this beautifully simple
        # expression: predicted probability minus the truth. Dividing by m averages
        # the blame over the batch.
        dz2 = (probs - Y_batch) / m

        # a1.T is the transpose (rows and columns swapped) so the shapes line up
        # for np.dot. This gives one gradient per weight in W2.
        dW2 = np.dot(a1.T, dz2) + lam * W2      # + lam * W2 is the L2 penalty: it pulls big weights toward 0
        db2 = dz2.sum(axis=0)                   # the bias gradient is just the sum of the errors (no L2 on biases)

        # Send the blame back to the hidden layer through W2...
        da1 = np.dot(dz2, W2.T)
        # ...and through ReLU: a neuron that output 0 had no effect, so it gets no blame.
        # (z1 > 0) is True/False, which NumPy treats as 1/0.
        dz1 = da1 * (z1 > 0)

        dW1 = np.dot(X_batch.T, dz1) + lam * W1  # gradients for the first layer's weights, plus L2
        db1 = dz1.sum(axis=0)                    # gradients for the first layer's biases

        # --- gradient descent: step every weight a little way downhill ---
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    # --- end of epoch: score the whole training set and the validation set ---
    _, _, train_probs = forward(X_train, W1, b1, W2, b2)
    train_loss = cross_entropy_loss(train_probs, Y_train)

    _, _, val_probs = forward(X_val, W1, b1, W2, b2)
    val_loss = cross_entropy_loss(val_probs, Y_val)

    train_losses.append(train_loss)     # .append adds one item to the end of a list
    val_losses.append(val_loss)

    # --- early stopping: remember the network from its best moment ---
    if val_loss < best_val_loss:                       # is this the lowest validation loss so far?
        best_val_loss = val_loss
        # .copy() takes a snapshot, so later training steps do not overwrite it.
        best_weights = (W1.copy(), b1.copy(), W2.copy(), b2.copy())
        best_epoch = epoch

    if epoch % 20 == 0 or epoch == epochs - 1:         # print a progress line now and then
        print(f"epoch {epoch:3d}  train loss {train_loss:.4f}  val loss {val_loss:.4f}")

# Throw away whatever the network drifted into at the end and restore the best version.
W1, b1, W2, b2 = best_weights
print(f"Best validation loss {best_val_loss:.4f} at epoch {best_epoch}")


# ---------------------------------------------------------------------
# STEP 6. THE FINAL, HONEST TEST
# ---------------------------------------------------------------------

# Open the sealed test set, once.
_, _, test_probs = forward(X_test, W1, b1, W2, b2)

# np.argmax returns the POSITION of the largest number in each row, so it turns
# probabilities like [0.02, 0.95, 0.03] into the network's answer: 1.
predictions = np.argmax(test_probs, axis=1)

# (predictions == y_test) gives a list of True/False. NumPy counts True as 1 and
# False as 0, so the mean of that list is exactly the fraction we got right.
accuracy = np.mean(predictions == y_test)
print(f"Test accuracy: {accuracy * 100:.1f}%  ({int(accuracy * len(y_test))} of {len(y_test)} wines correct)")

# The confusion matrix: rows are the true cultivar, columns are what we predicted.
# Counts on the diagonal are correct; anything off the diagonal is a mix-up.
confusion = np.zeros((3, 3), dtype=int)                  # start with a 3x3 table of zeros
for true_label, predicted_label in zip(y_test, predictions):   # zip walks both lists side by side
    confusion[true_label, predicted_label] += 1          # add 1 to the matching cell

print("Confusion matrix (rows = true cultivar, columns = predicted):")
print(confusion)
