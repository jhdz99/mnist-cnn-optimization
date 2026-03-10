# CNN Model Optimization — MNIST

A deep learning project focused on building and optimizing a Convolutional Neural Network (CNN) for handwritten digit classification on the MNIST dataset using TensorFlow and Keras.

## Project Description

This project demonstrates the design, training, and optimization of a **Convolutional Neural Network (CNN)** for handwritten digit classification using the **MNIST dataset**.

The goal was to improve performance over a baseline CNN by experimenting with network architecture and training strategies. The optimized model incorporates deeper convolutional layers, batch normalization, dropout regularization, and adaptive learning rate scheduling.

To better understand model behavior, the project includes visualizations of:

- Training and validation learning curves
- Confusion matrix for classification performance
- Sample digit predictions
- Intermediate CNN layer activations

These visualizations help diagnose overfitting, evaluate model accuracy, and interpret how convolutional layers extract features from handwritten digits.

The optimized model achieves **~99% classification accuracy on the MNIST test dataset**, demonstrating the effectiveness of architecture tuning and training optimization.

This project showcases practical skills in **deep learning model development, performance optimization, and ML visualization using Python and TensorFlow.**

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- scikit-learn
- Seaborn

## Skills Demonstrated

- Convolutional Neural Networks (CNNs)
- Deep Learning with TensorFlow & Keras
- Hyperparameter and architecture optimization
- Model evaluation and performance analysis
- Visualization of training metrics and neural network activations
- Data preprocessing for image classification
- Python-based machine learning workflows****

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
