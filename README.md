# MLP-DigitClassifier-Java

A Multi-Layer Perceptron (MLP) implemented in Java for binary classification of handwritten digits using the UCI Optical Recognition of Handwritten Digits dataset. This project demonstrates the power of MLPs in solving non-linear classification problems by achieving high accuracy with Two-Fold Cross-Validation.

---

## Features
- **Custom MLP Implementation**: Forward propagation, backpropagation, and gradient descent implemented from scratch.
- **Binary Classification**: Classifies whether a digit is `0` or not (`1` for `0` and `-1` for others).
- **Two-Fold Cross-Validation**: Ensures robust evaluation of the model's performance.
- **Feature Normalization**: Scales input features to improve numerical stability during training.

---

## Dataset
The project uses the [UCI Optical Recognition of Handwritten Digits Dataset](https://archive.ics.uci.edu/ml/datasets/optical+recognition+of+handwritten+digits), which contains:
- **Features**: 64 features representing pixel intensities of 8x8 grayscale images.
- **Labels**: Binary classification labels (`1` for digit `0`, `-1` for other digits).

---

## Installation

### Prerequisites
- Java Development Kit (JDK) 8 or higher
- An IDE (e.g., Eclipse, IntelliJ IDEA) or command-line tools for Java

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Akshar246/MLP-UCI-Digit-Classifier.git
