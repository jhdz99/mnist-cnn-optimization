import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model

os.makedirs("outputs", exist_ok=True)


def load_data():
    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = x_test.astype("float32") / 255.0
    x_test = np.expand_dims(x_test, axis=-1)
    return x_test, y_test


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()


def plot_sample_predictions(model, x_test, y_test, num_samples=10):
    predictions = model.predict(x_test[:num_samples], verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    plt.figure(figsize=(15, 4))
    for i in range(num_samples):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_test[i].squeeze(), cmap="gray")
        plt.title(f"True: {y_test[i]}\nPred: {predicted_labels[i]}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("outputs/sample_predictions.png")
    plt.close()


def main():
    model = load_model("models/best_mnist_cnn.keras")
    x_test, y_test = load_data()

    predictions = model.predict(x_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)

    test_acc = np.mean(y_pred == y_test)
    print(f"Test Accuracy: {test_acc:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    plot_confusion_matrix(y_test, y_pred)
    plot_sample_predictions(model, x_test, y_test)


if __name__ == "__main__":
    main()