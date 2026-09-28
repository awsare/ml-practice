# Fashion MNIST Autoencoder

A fully connected autoencoder built with PyTorch and trained on the Fashion MNIST dataset. This project demonstrates unsupervised learning, encoder-decoder architectures, latent-space compression, mean squared error (MSE) reconstruction loss, and the Adam optimizer.

## Run

Install the required dependencies:

```bash
pip install torch torchvision matplotlib
```

Run the autoencoder from this folder:

```bash
python autoencoder.py
```

The program downloads Fashion MNIST into a `data` folder, trains the autoencoder for 20 epochs to compress each 784-pixel image into a 64-dimensional latent vector and reconstruct it, and saves a comparison of original and reconstructed test images to `autoencoder_reconstructions.png`.
