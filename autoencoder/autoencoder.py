import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Task 1: Data Preparation
# ToTensor converts each 28x28 image to a float tensor of shape (1, 28, 28) with values in [0.0, 1.0]
transform = transforms.ToTensor()

train_dataset = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)

# Each batch is (images, labels); the autoencoder only uses the images and ignores the labels
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

images, _ = next(iter(train_loader))
print(f"Train: {len(train_dataset)} images, test: {len(test_dataset)} images")
print(f"Batch shape: {tuple(images.shape)}, pixel range: [{images.min():.1f}, {images.max():.1f}]")


# Task 2: Autoencoder Architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: 784 -> 128 -> 64 (the 64-dim output is the latent "bottleneck")
        self.encoder = nn.Sequential(
            nn.Linear(784, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        # Decoder: 64 -> 128 -> 784; Sigmoid keeps outputs in [0, 1] like the input pixels
        self.decoder = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 784), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


model = Autoencoder()
print(model)
recon = model(images.view(images.size(0), -1))
print(f"Reconstruction shape: {tuple(recon.shape)}, range: [{recon.min():.2f}, {recon.max():.2f}]")


# Task 3: Model Training
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    model.train()
    running_loss = 0.0
    for images, _ in train_loader:                   # labels are not needed
        x = images.view(images.size(0), -1)          # flatten 28x28 -> 784
        recon = model(x)                             # forward pass
        loss = criterion(recon, x)                   # compare reconstruction to the original
        optimizer.zero_grad()                        # zero the gradients
        loss.backward()                              # backward pass
        optimizer.step()                             # update the weights
        running_loss += loss.item() * x.size(0)
    print(f"Epoch {epoch+1}/20: train MSE={running_loss / len(train_dataset):.4f}")


# Task 4: Visualizing Reconstructions
model.eval()
images, _ = next(iter(test_loader))                 # a batch of test images
with torch.no_grad():
    recon = model(images.view(images.size(0), -1))

# Top row: 2 original test images; bottom row: their reconstructions directly below
fig, axes = plt.subplots(2, 2, figsize=(5, 5))
for col in range(2):
    axes[0, col].imshow(images[col].squeeze(), cmap="gray")
    axes[0, col].set_title(f"Original {col+1}")
    axes[1, col].imshow(recon[col].view(28, 28), cmap="gray")
    axes[1, col].set_title(f"Reconstruction {col+1}")
for ax in axes.flat:
    ax.axis("off")
fig.tight_layout()
fig.savefig("autoencoder_reconstructions.png", dpi=150)
plt.show()
