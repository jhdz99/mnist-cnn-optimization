# CNN Model Optimization — MNIST

A deep learning project focused on building and optimizing a Convolutional Neural Network (CNN) for handwritten digit classification on the MNIST dataset using TensorFlow and Keras.

## Project Overview

This project demonstrates the full workflow of:
- Building a baseline CNN for image classification
- Improving performance through architecture and hyperparameter optimization
- Evaluating model performance with classification metrics and confusion matrices
- Visualizing training curves and intermediate layer activations for model diagnosis

The optimized CNN achieved approximately **99%+ test accuracy** on MNIST.

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- scikit-learn
- Seaborn

## Features

- Baseline CNN vs optimized CNN comparison
- Early stopping and learning rate reduction
- Training/validation accuracy and loss visualization
- Confusion matrix and classification report
- Sample digit predictions
- Layer activation visualization for CNN interpretability

## Project Structure

```text
mnist-cnn-optimization/
│
├── README.md
├── requirements.txt
├── train.py
├── evaluate.py
├── visualize.py
├── utils.py
├── models/
│   └── best_mnist_cnn.keras
└── outputs/
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── sample_predictions.png
    └── activations.png