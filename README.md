# Neural Network Classifier — Non-linear Decision Boundaries

3-layer feedforward neural network for binary classification, built entirely from matrix operations. Demonstrates why neural networks outperform linear classifiers when decision boundaries are non-linear.

## Problem

Logistic regression produces a linear decision boundary and fails on data with curved or overlapping class regions. A shallow neural network with non-linear activations resolves this with a small number of parameters.

## Implementation

- **Architecture:** input → hidden layer (tanh activation) → output (softmax)
- **Training:** gradient descent with configurable learning rate and iteration count
- **Evaluation:** decision boundary visualization and accuracy comparison against logistic regression baseline
- **Two implementations:** functional (`nn_from_scratch.py`) and class-based (`ann_classification.py`) — same model, different code organization

## Stack

`Python` `NumPy` `scikit-learn` `matplotlib`
