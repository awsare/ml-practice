import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import wandb

def to_one_hot(labels):
    """Convert integer labels to one-hot encoding; returns one-hot array."""
    num_classes = len(set(labels))
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot

def softmax(z):
    """Compute row-wise softmax; returns probability array."""
    # BEGIN YOUR CODE HERE (~1-3 lines)
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    proba = np.exp(z_shifted) / np.sum(np.exp(z_shifted), axis=1, keepdims=True)
    # END YOUR CODE HERE
    return proba

def y_hat(W, b, images):
    """Compute predictions for images; returns probability array."""

    z = np.dot(images, W) + b
    predictions = softmax(z)

    return predictions


def cross_entropy_loss(W, b, images, labels, reg_strength=0.0):
    """Compute cross-entropy loss for predictions, optionally L2-regularized on W; returns scalar loss."""
    # BEGIN YOUR CODE HERE (~2-6 lines)
    probs = y_hat(W, b, images)

    loss = -np.sum(labels * np.log(probs)) / images.shape[0]
    loss += reg_strength * np.sum(W * W)
    # END YOUR CODE HERE
    return loss

def compute_gradient(W, b, images, labels, reg_strength=0.0):
    """Compute gradients w.r.t. weights and bias, with L2 regularization on W only; returns (dW, db)."""
    # BEGIN YOUR CODE HERE (~4-7 lines)
    probs = y_hat(W, b, images)

    error = probs - labels
    dW = np.dot(images.T, error) / images.shape[0] + 2 * reg_strength * W
    db = np.mean(error, axis=0)
    # END YOUR CODE HERE
    return dW, db

def compute_accuracy(W, b, images, labels):
    """Compute classification accuracy; returns fraction correct."""
    # BEGIN YOUR CODE HERE (~2-5 lines)
    probs = y_hat(W, b, images)

    predicted_labels = np.argmax(probs, axis=1)
    true_labels = np.argmax(labels, axis=1)
    acc = np.mean(predicted_labels == true_labels)
    # END YOUR CODE HERE
    return acc
            
def show_weights(W):
    """Render weights as image patches; returns None."""
    img_size = int(W.shape[0] ** 0.5)
    canvas = np.zeros((img_size, img_size * W.shape[1]))
    for idx, col in enumerate(range(0, canvas.shape[1], img_size)):
        canvas[:, col:col+img_size] = np.reshape(W[:-1, idx], (img_size, img_size))
    plt.imshow(canvas, cmap='gray')
    plt.show()

def train_softmax_classifier(train_images, train_labels, val_images, val_labels,
                             learning_rate=1e-5, batch_size=16, nepochs=100, reg_strength=0.0):
    """Train softmax classifier; returns (W, b, loss, accuracy)."""
    num_batches = train_images.shape[0] // batch_size
    n_features = train_images.shape[1]
    n_classes = train_labels.shape[1]

    # Initialize weights and bias
    # BEGIN YOUR CODE HERE (~2 lines)
    W = np.random.randn(n_features, n_classes) * 0.01
    b = np.zeros(n_classes)
    # END YOUR CODE HERE

    # fixed subsample of the training set for cheap per-epoch train-metric logging
    subsample_size = min(2000, train_images.shape[0])
    subsample_idx = np.random.RandomState(0).choice(train_images.shape[0], subsample_size, replace=False)

    cost, acc = float('inf'), 0.0
    for epoch in range(nepochs):
        np.random.seed(epoch)
        perm = np.random.permutation(train_images.shape[0])
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        for batch in range(num_batches):
            # what do you need to do in each iteration?
            # BEGIN YOUR CODE HERE (~5 lines)
            start = batch * batch_size
            end = start + batch_size
            batch_images = train_images[start:end]
            batch_labels = train_labels[start:end]
            dW, db = compute_gradient(W, b, batch_images, batch_labels, reg_strength=reg_strength)
            W -= learning_rate * dW
            b -= learning_rate * db
            # END YOUR CODE HERE
        # what do you need to compute at the end of each epoch?
        # BEGIN YOUR CODE HERE (~3 lines)
        cost = cross_entropy_loss(W, b, val_images, val_labels, reg_strength=reg_strength)
        acc = compute_accuracy(W, b, val_images, val_labels)
        # END YOUR CODE HERE
        train_loss = cross_entropy_loss(W, b, train_images[subsample_idx], train_labels[subsample_idx],
                                         reg_strength=reg_strength)
        train_acc = compute_accuracy(W, b, train_images[subsample_idx], train_labels[subsample_idx])
        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'train_accuracy': train_acc,
                   'val_loss': cost, 'val_accuracy': acc})
    return W, b, cost, acc

# def add_bias(images):
#     return np.hstack((images, np.ones((images.shape[0], 1))))

if __name__ == "__main__":
    # Load data
    training_images = np.load("fashion_mnist_train_images.npy") / 255.0 - 0.5
    training_labels = to_one_hot(np.load("fashion_mnist_train_labels.npy"))
    testing_images = np.load("fashion_mnist_test_images.npy") / 255.0 - 0.5
    testing_labels = to_one_hot(np.load("fashion_mnist_test_labels.npy"))

    # split training into training and validation using train_test_split
    x_train, x_val, y_train, y_val = train_test_split(training_images, training_labels,
                                                      test_size=0.2, random_state=541)

    # List hyperparameters to try
    # BEGIN YOUR CODE HERE (~2-3 lines)
    learning_rates = [1e-3, 1e-4, 1e-5]
    reg_strengths = [0.0, 1e-4, 1e-3]
    batch_size = 32
    # END YOUR CODE HERE

    # Initialize variables to keep track of best hyperparameters
    # BEGIN YOUR CODE HERE (~3 lines)
    best_lr, best_reg = learning_rates[0], reg_strengths[0]
    best_acc = 0.0
    # END YOUR CODE HERE

    # Train model
    # BEGIN YOUR CODE HERE (~7-10 lines)
    for lr in learning_rates:
        for reg in reg_strengths:
            wandb.init(project="cs541-hw2-fashion-mnist",
                       config={"learning_rate": lr, "reg_strength": reg, "batch_size": batch_size},
                       group="lr_reg_sweep")

            print(f"Training with learning rate: {wandb.config.learning_rate}, "
                  f"regularization strength: {wandb.config.reg_strength}")
            W, b, cost, acc = train_softmax_classifier(x_train, y_train, x_val, y_val,
                                                        learning_rate=wandb.config.learning_rate,
                                                        batch_size=wandb.config.batch_size,
                                                        nepochs=100, reg_strength=wandb.config.reg_strength)

            if acc > best_acc:
                best_acc = acc
                best_lr, best_reg = wandb.config.learning_rate, wandb.config.reg_strength

            print(f"Validation accuracy: {acc:.4f} with learning rate: {wandb.config.learning_rate}, "
                  f"regularization strength: {wandb.config.reg_strength}")
            wandb.finish()
    # END YOUR CODE HERE

    # Retrain model on full training set with best hyperparameters and evaluate on test set
    # BEGIN YOUR CODE HERE (~1 line)
    wandb.init(project="cs541-hw2-fashion-mnist",
               config={"learning_rate": best_lr, "reg_strength": best_reg, "batch_size": batch_size},
               group="lr_reg_sweep", tags=["final_eval"])
    final_W, final_b, loss, acc = train_softmax_classifier(training_images, training_labels, testing_images, testing_labels,
                                   learning_rate=best_lr, batch_size=batch_size, nepochs=100, reg_strength=best_reg)
    unreg_loss = cross_entropy_loss(final_W, final_b, testing_images, testing_labels)
    wandb.log({"test_loss": unreg_loss, "test_accuracy": acc})
    wandb.finish()
    print(f"Final test cross-entropy loss (unregularized): {unreg_loss:.4f}")
    print(f"Final test accuracy: {acc:.4f} with learning rate: {best_lr}, regularization strength: {best_reg}")
    # END YOUR CODE HERE

    # showWeights(lr[:,:])