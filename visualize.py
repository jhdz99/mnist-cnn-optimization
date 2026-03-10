import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model, Model

os.makedirs("outputs", exist_ok=True)


def load_sample():
    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = x_test.astype("float32") / 255.0
    x_test = np.expand_dims(x_test, axis=-1)
    return x_test, y_test


def visualize_activations(model, image, save_path="outputs/activations.png"):
    # Make sure the model has been called at least once
    _ = model(np.expand_dims(image, axis=0), training=False)

    layer_outputs = []
    layer_names = []

    for layer in model.layers:
        if "conv2d" in layer.name:
            layer_outputs.append(layer.output)
            layer_names.append(layer.name)

    if not layer_outputs:
        raise ValueError("No Conv2D layers found in the model.")

    activation_model = Model(inputs=model.inputs, outputs=layer_outputs)
    activations = activation_model.predict(np.expand_dims(image, axis=0), verbose=0)

    num_layers = len(activations)
    num_filters_to_show = 6

    fig, axes = plt.subplots(num_layers, num_filters_to_show, figsize=(12, 2 * num_layers))

    if num_layers == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, activation in enumerate(activations):
        num_filters = min(num_filters_to_show, activation.shape[-1])

        for col in range(num_filters_to_show):
            ax = axes[row, col]

            if col < num_filters:
                ax.imshow(activation[0, :, :, col], cmap="viridis")
                if col == 0:
                    ax.set_title(layer_names[row])
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    model = load_model("models/best_mnist_cnn.keras")
    x_test, y_test = load_sample()

    index = 0
    image = x_test[index]
    print(f"Visualizing activations for test image with label: {y_test[index]}")
    visualize_activations(model, image)
    print("Saved activation visualization to outputs/activations.png")


if __name__ == "__main__":
    main()