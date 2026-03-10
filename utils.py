import matplotlib.pyplot as plt


def plot_training_history(baseline_history, optimized_history, save_path="outputs/training_curves.png"):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(baseline_history.history["accuracy"], label="Baseline Train")
    plt.plot(baseline_history.history["val_accuracy"], label="Baseline Val")
    plt.plot(optimized_history.history["accuracy"], label="Optimized Train")
    plt.plot(optimized_history.history["val_accuracy"], label="Optimized Val")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(baseline_history.history["loss"], label="Baseline Train")
    plt.plot(baseline_history.history["val_loss"], label="Baseline Val")
    plt.plot(optimized_history.history["loss"], label="Optimized Train")
    plt.plot(optimized_history.history["val_loss"], label="Optimized Val")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()