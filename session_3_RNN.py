# KEY OBSERVATION: SEQUENCES - SUCH AS SEQUENCE OF WORDS

# EXAMPLE: I FLEW FROM HEATHROW AIRPORT
# here notice that words alone have little context
# Heathrow is a county in England
# Airport is where planes land
# Flew ~ Flying: is an action
# When considering the sequence of words however we have context, meaning
# That is what RNNs strive for when learning

# Install dependencies (run once):
#  pip install numpy pandas matplotlib tensorflow keras

import collections

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences


def plot_history(history, title=""):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(history.history["loss"], label="loss")
    ax1.plot(history.history["val_loss"], label="val_loss")
    ax1.legend()
    ax2.plot(history.history["accuracy"], label="acc")
    ax2.plot(history.history["val_accuracy"], label="val_acc")
    ax2.legend()
    fig.suptitle(title)
    plt.show()


# Load IMDB data.
(train_x, train_y), (test_x, test_y) = imdb.load_data(num_words=500)
print("Train/test set sizes:", len(train_x), len(test_x))

# Take a look at train_x and train_y see how documents are represented,
# and each document contains a different number of words.
# Note: At this point we don't have a vocabulary file
# so we don't actually know what these numbers constitute to.
# But they are word_IDs.
train_counts = collections.Counter(train_y)
print(f"Train label distribution: {train_counts[0]} negative, {train_counts[1]} positive")
test_counts = collections.Counter(test_y)
print(f"Test label distribution:  {test_counts[0]} negative, {test_counts[1]} positive")

# Padding sequences.
train_x = pad_sequences(train_x, maxlen=200)
test_x = pad_sequences(test_x, maxlen=200)

# --- Model 1 ---
print("\n" + "-" * 50)
print("Model 1")
print("-" * 50)

model = keras.Sequential([
    layers.Embedding(input_dim=500, output_dim=32),
    layers.SimpleRNN(units=8),  # default activation: tanh, S-shaped [-1..1].
    layers.Dense(units=1, activation="sigmoid")
])

model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Fit model.
history = model.fit(
    train_x, train_y,
    epochs=10,
    batch_size=128,
    validation_split=0.2
)

# 32 x 500 = 16000
# 328 = 8 x (8+32) + 8
model.summary()

plot_history(history, "Model 1")

# Prediction.
model.evaluate(train_x, train_y)
pred = model.predict(train_x).flatten()
print(pd.crosstab(pred, train_y, rownames=["Predicted"], colnames=["Actual"]))

model.evaluate(test_x, test_y)
pred1 = model.predict(test_x).flatten()
print("--- Test Predictions ---")
print(pd.crosstab(pred1, test_y, rownames=["Predicted"], colnames=["Actual"]))

# Possible Updates:
# - Number of units in RNN layer.
# - Different Activation Functions on RNN layer.
# - More RNN layers, obviously.
# - Length of Padding.

# --- Model 2 ---
print("\n" + "-" * 50)
print("Model 2")
print("-" * 50)

model = keras.Sequential([
    layers.Embedding(input_dim=500, output_dim=32),
    layers.SimpleRNN(units=32),  # default activation function is tanh.
    layers.Dense(units=1, activation="sigmoid")
])

model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Fit model.
history = model.fit(
    train_x, train_y,
    epochs=10,
    batch_size=128,
    validation_split=0.2
)

plot_history(history, "Model 2")

# Prediction.
model.evaluate(train_x, train_y)
pred = model.predict(train_x)
model.evaluate(test_x, test_y)
pred1 = model.predict(test_x)

# --- Model 3 ---
print("\n" + "-" * 50)
print("Model 3")
print("-" * 50)

model = keras.Sequential([
    layers.Embedding(input_dim=500, output_dim=32),
    layers.SimpleRNN(units=32, activation="relu"),
    layers.Dense(units=1, activation="sigmoid")
])

model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Fit model.
history = model.fit(
    train_x, train_y,
    epochs=10,
    batch_size=128,
    validation_split=0.2
)

plot_history(history, "Model 3")

# Prediction.
model.evaluate(train_x, train_y)
pred = model.predict(train_x)
model.evaluate(test_x, test_y)
pred1 = model.predict(test_x)

# --- Model 4 ---
print("\n" + "-" * 50)
print("Model 4")
print("-" * 50)

model = keras.Sequential([
    layers.Embedding(input_dim=500, output_dim=32),
    layers.SimpleRNN(units=32, return_sequences=True, activation="relu"),
    layers.SimpleRNN(units=32, return_sequences=True, activation="relu"),
    layers.SimpleRNN(units=32, activation="relu"),
    layers.Dense(units=1, activation="sigmoid")
])

model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Fit model.
history = model.fit(
    train_x, train_y,
    epochs=10,
    batch_size=128,
    validation_split=0.2
)

plot_history(history, "Model 4")

# Prediction.
model.evaluate(train_x, train_y)
model.evaluate(test_x, test_y)
