import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# -------------------------------
# Robust Differentiable SVD
# -------------------------------

class RobustSVD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, K=9, eps=1e-2):
        # Compute full SVD; we assume A is a square matrix (or batched square)
        # Here we use torch.linalg.svd which returns U, S, Vh (V transposed)
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        ctx.save_for_backward(U, S, Vh)
        ctx.K = K
        ctx.eps = eps
        return U, S, Vh

    @staticmethod
    def backward(ctx, grad_U, grad_S, grad_Vh):
        # Retrieve saved tensors and hyperparameters.
        U, S, Vh = ctx.saved_tensors
        K = ctx.K
        eps = ctx.eps

        # We assume A is batched: shape (B, m, m)
        # Add eps to S to avoid division by zero
        s = S + eps  # shape: (B, r)
        # Expand s to matrices for division: s_i in column and s_j in row.
        s_col = s.unsqueeze(2)  # (B, r, 1)
        s_row = s.unsqueeze(1)  # (B, 1, r)

        # Compute ratio s_j/s_i (for each batch)
        ratio = s_row / s_col  # shape: (B, r, r)
        # Compute the Taylor series expansion: sum_{k=0}^{K} (ratio)^k
        series = torch.zeros_like(ratio)
        for k in range(K + 1):
            series = series + ratio.pow(k)
        # Compute robust factor: (1/s_i)*series, but we only need off-diagonals.
        robust_K = (1.0 / s_col) * series  # shape: (B, r, r)
        # Set diagonal elements to 0 (they are not used)
        batch_size, r, _ = robust_K.shape
        eye = torch.eye(r, device=robust_K.device).unsqueeze(0).expand(batch_size, r, r)
        robust_K = robust_K * (1 - eye)

        # Now, following the derivations in Robust Differentiable SVD,
        # the gradient with respect to A is approximated by:
        # grad_A = U [diag(grad_S) + F] Vh
        # where F = robust_K ∘ (U^T grad_U - grad_Vh^T V)
        V = Vh.transpose(-2, -1)
        UT_gradU = torch.matmul(U.transpose(-2, -1), grad_U)  # (B, r, r)
        # For grad_Vh, note that grad_V = - (grad_Vh)^T.
        grad_V = -grad_Vh.transpose(-2, -1)  # (B, m, m)
        grad_V_term = torch.matmul(grad_V.transpose(-2, -1), V)  # (B, r, r)
        # According to our derivation, let X = UT_gradU + grad_V_term.
        X = UT_gradU + grad_V_term  # (B, r, r)
        # Multiply element-wise with robust_K
        F_term = robust_K * X  # (B, r, r)
        # Build the full term inside the bracket: diag(grad_S) + F_term.
        grad_S_diag = torch.diag_embed(grad_S)  # (B, r, r)
        inner = grad_S_diag + F_term  # (B, r, r)
        # Compute grad_A = U * inner * Vh.
        grad_A = torch.matmul(torch.matmul(U, inner), Vh)
        return grad_A, None, None


def robust_svd(A, K=9, eps=1e-2):
    # Wrapper function for RobustSVD
    return RobustSVD.apply(A, K, eps)


def robust_polar(X, K=9, eps=1e-2):
    """
    Computes the polar factor of X via its SVD.
    If X = U Σ V^T, then the polar factor is Q = U V^T.
    Here we use our robust differentiable SVD.
    """
    U, S, Vh = robust_svd(X, K, eps)
    Q = torch.matmul(U, Vh)
    return Q


# -------------------------------
# PyTorch Lightning Module
# -------------------------------

class SVDPredictor(pl.LightningModule):
    def __init__(self, m=28, n=28, hidden_dim=256):
        """
        Predicts the full SVD of an m x n matrix.
        Instead of using QR to enforce orthogonality, we use a robust
        differentiable SVD layer (via a robust polar decomposition) to
        obtain orthogonal matrices.
        """
        super().__init__()
        self.m = m
        self.n = n
        self.min_dim = min(m, n)
        input_dim = m * n

        # Common encoder network.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Head for predicting a raw matrix for U.
        self.fc_u = nn.Linear(hidden_dim, m * m)

        # Head for predicting singular values (min_dim parameters).
        self.fc_s = nn.Linear(hidden_dim, self.min_dim)

        # Head for predicting a raw matrix for V.
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
        A_flat = A.view(batch_size, -1)  # Flatten A.
        latent = self.encoder(A_flat)

        # Predict raw matrices and singular values.
        u_raw = self.fc_u(latent).view(batch_size, self.m, self.m)
        s_raw = self.fc_s(latent)  # (batch, min_dim)
        v_raw = self.fc_v(latent).view(batch_size, self.n, self.n)

        # Instead of QR, use robust polar decomposition to get orthogonal matrices.
        U = robust_polar(u_raw)  # (batch, m, m)
        V = robust_polar(v_raw)  # (batch, n, n)

        # Ensure singular values are nonnegative.
        s = F.softplus(s_raw)

        return U, s, V

    def training_step(self, batch, batch_idx):
        # 'batch' is a tuple (x, y) from MNIST, we only need the image.
        x, _ = batch
        # x has shape (batch, 1, 28, 28); squeeze to (batch, 28, 28)
        A = x.squeeze(1)
        U, s, V = self(A)

        # Build the diagonal matrix for s.
        S_mat = torch.zeros(A.size(0), self.m, self.n, device=A.device)
        for i in range(self.min_dim):
            S_mat[:, i, i] = s[:, i]

        # Reconstruct the matrix: A_pred = U * S * V^T.
        A_pred = torch.matmul(torch.matmul(U, S_mat), V.transpose(-2, -1))
        loss = F.mse_loss(A_pred, A)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        A = x.squeeze(1)
        U, s, V = self(A)
        S_mat = torch.zeros(A.size(0), self.m, self.n, device=A.device)
        for i in range(self.min_dim):
            S_mat[:, i, i] = s[:, i]
        A_pred = torch.matmul(torch.matmul(U, S_mat), V.transpose(-2, -1))
        loss = F.mse_loss(A_pred, A)
        self.log('val_loss', loss)

        if batch_idx % 100 == 0:
            print(f"Epoch {self.current_epoch}, Step {batch_idx}, Validation Loss: {loss.item()}")
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


# -------------------------------
# Dataset and Training
# -------------------------------

class SVDDataset(Dataset):
    def __init__(self, num_samples=1000, m=28, n=28):
        self.num_samples = num_samples
        self.m = m
        self.n = n

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a random matrix (e.g., a random MNIST-like image).
        A = torch.randn(self.m, self.n)
        return A


if __name__ == '__main__':
    # Define MNIST transforms.
    transform = transforms.Compose([
        transforms.ToTensor(),  # (1, 28, 28)
    ])

    # Prepare MNIST training dataset.
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True, num_workers=4)

    train_size = int(0.8 * len(mnist_train))
    val_size = len(mnist_train) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(mnist_train, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=4)


    # Initialize the model.
    model = SVDPredictor(m=28, n=28, hidden_dim=256)

    # Train the model.
    trainer = pl.Trainer(max_epochs=10, accelerator="auto", devices=1, fast_dev_run=False, enable_progress_bar=False)
    trainer.fit(model, train_loader, val_loader)

    # -------------------------------
    # Visualization on Test Data
    # -------------------------------
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(mnist_test, batch_size=10, shuffle=True, num_workers=4)
    model.eval()
    batch = next(iter(test_loader))
    images, _ = batch  # shape: (batch, 1, 28, 28)
    images = images.squeeze(1)  # shape: (batch, 28, 28)
    with torch.no_grad():
        U, s, V = model(images)
    S_mat = torch.zeros(images.size(0), 28, 28, device=images.device)
    for i in range(model.min_dim):
        S_mat[:, i, i] = s[:, i]
    reconstructions = torch.matmul(torch.matmul(U, S_mat), V.transpose(-2, -1))

    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    for i in range(10):
        axes[0, i].imshow(images[i].cpu().numpy(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstructions[i].cpu().numpy(), cmap='gray')
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel("Original", fontsize=14)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=14)
    plt.suptitle("Robust Differentiable SVD Prediction on MNIST", fontsize=16)
    plt.show()
    # https://arxiv.org/pdf/2104.03821
