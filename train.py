import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from utils import plot_training_history

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    return (x_train, y_train, y_train_cat), (x_test, y_test, y_test_cat)


def build_baseline_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_optimized_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),

        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_model(model, x_train, y_train_cat, model_name, epochs=15, batch_size=128):
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        ModelCheckpoint(
            filepath=f"models/{model_name}.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]

    history = model.fit(
        x_train,
        y_train_cat,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return history


def main():
    (x_train, y_train, y_train_cat), (x_test, y_test, y_test_cat) = load_data()

    print("\nTraining baseline model...")
    baseline = build_baseline_model()
    baseline_history = train_model(baseline, x_train, y_train_cat, "baseline_mnist_cnn", epochs=8)

    baseline_loss, baseline_acc = baseline.evaluate(x_test, y_test_cat, verbose=0)
    print(f"Baseline Test Accuracy: {baseline_acc:.4f}")

    print("\nTraining optimized model...")
    optimized = build_optimized_model()
    optimized_history = train_model(optimized, x_train, y_train_cat, "best_mnist_cnn", epochs=15)

    optimized_loss, optimized_acc = optimized.evaluate(x_test, y_test_cat, verbose=0)
    print(f"Optimized Test Accuracy: {optimized_acc:.4f}")

    plot_training_history(
        baseline_history,
        optimized_history,
        save_path="outputs/training_curves.png"
    )

    print("\nDone.")
    print(f"Baseline accuracy:  {baseline_acc:.4f}")
    print(f"Optimized accuracy: {optimized_acc:.4f}")


if __name__ == "__main__":
    main()