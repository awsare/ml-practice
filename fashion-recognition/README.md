# Fashion MNIST Classification

A Fashion MNIST image classification model built with NumPy. This project demonstrates softmax regression, one-hot encoding, cross-entropy loss, gradient descent, L2 regularization, and hyperparameter evaluation with Weights & Biases (W&B).

## Data

Download the following files into this folder before running the classifier:

- [Training images](https://s3.amazonaws.com/jrwprojects/fashion_mnist_train_images.npy)
- [Training labels](https://s3.amazonaws.com/jrwprojects/fashion_mnist_train_labels.npy)
- [Test images](https://s3.amazonaws.com/jrwprojects/fashion_mnist_test_images.npy)
- [Test labels](https://s3.amazonaws.com/jrwprojects/fashion_mnist_test_labels.npy)

## Run

Install the required dependencies:

```bash
pip install numpy matplotlib scikit-learn wandb
```

Log in to Weights & Biases:

```bash
wandb login
```

Run the classifier from this folder:

```bash
python fmnist_classification.py
```

The program loads the downloaded Fashion MNIST `.npy` files, trains a softmax classifier across several learning-rate and regularization combinations, and logs training and validation metrics to the `cs541-hw2-fashion-mnist` W&B project before evaluating the best model on the test set.