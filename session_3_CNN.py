# Install dependencies (run once):
#   pip install numpy matplotlib Pillow tensorflow
import os
import glob

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras import layers


# --- Load images ---

high_dir = "/Users/alinakatiliute/university/BIM RSM/Deep Learning Python/Data/high/"
low_dir = "/Users/alinakatiliute/university/BIM RSM/Deep Learning Python/Data/low/"

def load_images(directory):
    paths = sorted(glob.glob(os.path.join(directory, "*.jpg")))
    return [np.array(Image.open(p).convert("RGB")) for p in paths]


pics_high = load_images(high_dir)
pics_low = load_images(low_dir)

# Example: convert one image to grayscale and display it.
image = Image.open(sorted(glob.glob(os.path.join(low_dir, "*.jpg")))[0])
plt.imshow(image)
plt.title("Original")
plt.axis("off")
plt.show()

z_gray = image.convert("L")
plt.imshow(z_gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")
plt.show()

# --- Labels ---

objective = np.array([1] * 200 + [0] * 200)  # 1 = high, 0 = low.

# --- Merge and resize ---

all_images = pics_high + pics_low  # List of 400 numpy arrays.

TARGET_SIZE = (128, 128)


def resize_image(img_array):
    img = Image.fromarray(img_array).resize(TARGET_SIZE)
    return np.array(img)


picts = [resize_image(img) for img in all_images]

# --- Train / test split (80 / 20) ---

np.random.seed(42)
tf.random.set_seed(42)
indexes = np.random.choice([1, 2], size=len(picts), replace=True, p=[0.8, 0.2])

train_images = [picts[i] for i in range(len(picts)) if indexes[i] == 1]
test_images = [picts[i] for i in range(len(picts)) if indexes[i] == 2]
tr_obj = objective[indexes == 1]
te_obj = objective[indexes == 2]

# --- Stack into arrays and normalize ---

# Shape: (N, 128, 128, 3), pixels scaled to [0, 1].
train_corpus = np.stack(train_images).astype("float32") / 255.0
test_corpus = np.stack(test_images).astype("float32") / 255.0

# --- Plot a 6x6 grid of training images ---

fig, axes = plt.subplots(6, 6, figsize=(10, 10))
for idx, ax in enumerate(axes.flat):
    ax.imshow(train_corpus[idx])
    ax.axis("off")
plt.tight_layout()
plt.show()

# --- One-hot encode labels ---

train_target = keras.utils.to_categorical(tr_obj, num_classes=2)
test_target = keras.utils.to_categorical(te_obj, num_classes=2)

# --- Build CNN ---

model = keras.Sequential([
    layers.Conv2D(32, (2, 2), activation="relu", input_shape=(128, 128, 3)),
    layers.Conv2D(32, (2, 2), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.25),
    layers.Dense(2, activation="softmax"),
])

model.summary()

# --- Compile ---

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

# --- Train ---

history = model.fit(
    train_corpus, train_target,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
)

# --- Plot training history ---

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

ax1.plot(history.history["loss"], label="Train")
ax1.plot(history.history["val_loss"], label="Validation")
ax1.set_title("Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()

ax2.plot(history.history["accuracy"], label="Train")
ax2.plot(history.history["val_accuracy"], label="Validation")
ax2.set_title("Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.legend()

plt.suptitle("Training History", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# --- Evaluate and predict - train data ---

train_loss, train_acc = model.evaluate(train_corpus, train_target, verbose=0)
pred_train = model.predict(train_corpus)

# --- Evaluate and predict - test data ---

test_loss, test_acc = model.evaluate(test_corpus, test_target, verbose=0)
pred_test = model.predict(test_corpus)

# --- Print results ---

print(f"\nTrain - loss: {train_loss:.4f}  accuracy: {train_acc:.2%}")
print(f"Test  - loss: {test_loss:.4f}  accuracy: {test_acc:.2%}")
