import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class SVDPredictor(pl.LightningModule):
    def __init__(self, m=28, n=28, hidden_dim=256):
        """
        Predicts the full SVD of an m x n matrix.
        U_pred: m x m, s_pred: min(m, n), V_pred: n x n.
        """
        super().__init__()
        self.m = m
        self.n = n
        self.min_dim = min(m, n)
        input_dim = m * n

        # A simple encoder network.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Head for predicting U (raw parameters to be orthogonalized)
        self.fc_u = nn.Linear(hidden_dim, m * m)

        # Head for predicting singular values (min_dim parameters)
        self.fc_s = nn.Linear(hidden_dim, self.min_dim)

        # Head for predicting V (raw parameters to be orthogonalized)
        self.fc_v = nn.Linear(hidden_dim, n * n)

    def forward(self, A):
        """
        Forward pass for a batch of matrices A of shape (batch, m, n).
        Returns U, s, V where:
          U: (batch, m, m) orthogonal matrix,
          s: (batch, min_dim) non-negative singular values,
          V: (batch, n, n) orthogonal matrix.
        """
        batch_size = A.size(0)
        A_flat = A.view(batch_size, -1)  # Flatten the matrix

        latent = self.encoder(A_flat)

        # Get raw outputs for U, s, and V.
        u_raw = self.fc_u(latent).view(batch_size, self.m, self.m)
        s_raw = self.fc_s(latent)  # (batch, min_dim)
        v_raw = self.fc_v(latent).view(batch_size, self.n, self.n)

        # Project u_raw and v_raw to orthogonal matrices using QR decomposition.
        u, _ = torch.linalg.qr(u_raw)
        v, _ = torch.linalg.qr(v_raw)

        # Ensure the singular values are nonnegative.
        s = F.softplus(s_raw)

        return u, s, v

    def training_step(self, batch, batch_idx):
        # 'batch' is a tuple (x, y), but we only need the image.
        x, _ = batch
        # x has shape (batch, 1, 28, 28); remove the channel dimension.
        A = x.squeeze(1)
        u_pred, s_pred, v_pred = self(A)

        # Build a diagonal matrix from s_pred.
        S_pred = torch.zeros(A.size(0), self.m, self.n, device=A.device)
        for i in range(self.min_dim):
            S_pred[:, i, i] = s_pred[:, i]

        # Reconstruct the matrix: A_pred = U * S * V^T.
        A_pred = u_pred @ S_pred @ v_pred.transpose(-2, -1)

        # Reconstruction loss (MSE between original image and reconstruction).
        loss = F.mse_loss(A_pred, A)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        A = x.squeeze(1)
        u_pred, s_pred, v_pred = self(A)
        S_pred = torch.zeros(A.size(0), self.m, self.n, device=A.device)
        for i in range(self.min_dim):
            S_pred[:, i, i] = s_pred[:, i]
        A_pred = u_pred @ S_pred @ v_pred.transpose(-2, -1)
        loss = F.mse_loss(A_pred, A)
        self.log('val_loss', loss)

        if batch_idx % 100 == 0:
            print(f"Epoch {self.current_epoch}, Step {batch_idx}, Validation Loss: {loss.item()}")
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


if __name__ == '__main__':
    # Define transforms for MNIST.
    transform = transforms.Compose([
        transforms.ToTensor(),  # MNIST images are converted to [0,1] tensors of shape (1, 28, 28)
    ])

    # Prepare the MNIST training dataset and dataloader.
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(mnist_train, batch_size=64, shuffle=True, num_workers=4)
    # split the data into training and validation sets
    train_size = int(0.8 * len(mnist_train))
    val_size = len(mnist_train) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(mnist_train, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=4)



    # Initialize the model.
    model = SVDPredictor(m=28, n=28, hidden_dim=256)

    # Create a PyTorch Lightning trainer and train the model.
    trainer = pl.Trainer(max_epochs=10, accelerator="auto", devices=1, fast_dev_run=False, enable_progress_bar=False)
    trainer.fit(model, train_loader, val_loader)

    # After training, prepare the MNIST test dataset for visualization.
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(mnist_test, batch_size=10, shuffle=True, num_workers=4)

    # Get one batch of test images.
    model.eval()  # Set the model to evaluation mode.
    batch = next(iter(test_loader))
    images, _ = batch  # images shape: (batch, 1, 28, 28)
    images = images.squeeze(1)  # Now shape: (batch, 28, 28)

    with torch.no_grad():
        U, s, V = model(images)

    # Build the diagonal matrix for each sample.
    S_pred = torch.zeros(images.size(0), 28, 28, device=images.device)
    for i in range(model.min_dim):
        S_pred[:, i, i] = s[:, i]

    # Reconstruct images using the predicted SVD.
    reconstructions = U @ S_pred @ V.transpose(-2, -1)

    # Plot the original and reconstructed images side by side.
    fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(20, 4))
    for i in range(10):
        axes[0, i].imshow(images[i].cpu().numpy(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstructions[i].cpu().numpy(), cmap='gray')
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel("Original", fontsize=14)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=14)
    plt.suptitle("SVD Prediction on MNIST", fontsize=16)
    plt.show()
