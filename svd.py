import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from rich.jupyter import display
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import torch.linalg as linalg  # for SVD

torch.set_float32_matmul_precision('medium')


# --------------------------
# Dataset: CIFAR10 with SVD targets
# --------------------------
class CIFAR10SVDDataset(Dataset):
    def __init__(self, train=True, transform=None, k_components=10):
        """
        For each grayscale CIFAR-10 image (assumed to be 32x32), we compute
        the top k singular triplets from its SVD decomposition.
        Each token is the concatenation of [u (32d), sigma (1d), v (32d)].
        """
        self.dataset = datasets.CIFAR10(
            root='data', train=train, download=True, transform=transform
        )
        self.k_components = k_components

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get image and label (we ignore the label)
        img, _ = self.dataset[idx]
        # img is expected to be grayscale with shape [1, 32, 32]
        # Remove channel dimension -> [32, 32]
        img_gray = img.squeeze(0)

        # Compute SVD. We use torch.linalg.svd (full_matrices=False gives U: [32,32], S: [32], V: [32,32])
        U, S, V = linalg.svd(img_gray, full_matrices=False)
        k = self.k_components
        # We want the top k components.
        # For each singular triplet i: u_i = U[:, i], sigma = S[i], v_i = V[i, :]
        # For convenience, we transpose U so that each row corresponds to one component.
        U_k = U[:, :k].transpose(0, 1)  # shape: [k, 32]
        S_k = S[:k].unsqueeze(1)  # shape: [k, 1]
        V_k = V[:k, :]  # shape: [k, 32]

        # Enforce a simple sign convention: ensure the first element of u is nonnegative.
        for i in range(k):
            if U_k[i, 0] < 0:
                U_k[i] = -U_k[i]
                V_k[i] = -V_k[i]

        # Concatenate to form the target token for each component: [u, sigma, v]
        target_seq = torch.cat([U_k, S_k, V_k], dim=1)  # shape: [k, 65]
        # Return the original image (for conditioning) and the SVD token sequence.
        return img, target_seq


# --------------------------
# Lightning Module: SVD Autoregressive Model
# --------------------------
class SVDAutoRegressiveModel(pl.LightningModule):
    def __init__(self, latent_dim=128, k_components=10, image_size=32):
        super().__init__()
        self.k_components = k_components
        self.image_size = image_size
        # Each token consists of: u (image_size-dim), sigma (1-dim), v (image_size-dim)
        self.token_dim = image_size + 1 + image_size  # 32 + 1 + 32 = 65

        # Weight for image reconstruction loss
        self.lambda_recon = 0.1

        # --- Encoder ---
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 32x16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64x8x8
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 128x4x4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, latent_dim),
            nn.ReLU(),
        )

        # --- Decoder ---
        self.decoder_embedding = nn.Linear(self.token_dim, latent_dim)
        self.lstm = nn.LSTM(latent_dim, latent_dim, batch_first=True)
        self.output_layer = nn.Linear(latent_dim, self.token_dim)

        # A learned start token (initial input to the LSTM decoder)
        self.start_token = nn.Parameter(torch.zeros(1, self.token_dim))
        self.latent_to_hidden = nn.Linear(latent_dim, latent_dim)
        self.latent_to_cell = nn.Linear(latent_dim, latent_dim)

        # Loss function: MSE between predicted and target tokens.
        self.criterion = nn.MSELoss()

    def forward(self, x, target_seq=None, teacher_forcing_ratio=0.5):
        """
        x: input image tensor of shape (batch, 1, H, W)
        target_seq: ground truth token sequence of shape (batch, k_components, token_dim)
        teacher_forcing_ratio: probability of using the ground truth token at each step.
        """
        batch_size = x.size(0)
        # Encode the image to a latent representation.
        latent = self.encoder(x)  # shape: (batch, latent_dim)

        # Initialize LSTM hidden state (h0, c0) from latent vector.
        h0 = self.latent_to_hidden(latent).unsqueeze(0)  # shape: (1, batch, latent_dim)
        c0 = self.latent_to_cell(latent).unsqueeze(0)  # shape: (1, batch, latent_dim)

        outputs = []
        # Expand the learned start token for the entire batch.
        input_token = self.start_token.expand(batch_size, -1)  # shape: (batch, token_dim)
        seq_len = self.k_components

        for t in range(seq_len):
            # Embed the input token.
            embed = self.decoder_embedding(input_token).unsqueeze(1)  # shape: (batch, 1, latent_dim)
            # One step of the LSTM.
            lstm_out, (h0, c0) = self.lstm(embed, (h0, c0))  # lstm_out: (batch, 1, latent_dim)
            token_pred = self.output_layer(lstm_out.squeeze(1))  # shape: (batch, token_dim)
            outputs.append(token_pred.unsqueeze(1))

            # Decide whether to use teacher forcing.
            if self.training and target_seq is not None and torch.rand(1).item() < teacher_forcing_ratio:
                input_token = target_seq[:, t, :]
            else:
                input_token = token_pred

        outputs = torch.cat(outputs, dim=1)  # shape: (batch, seq_len, token_dim)
        return outputs

    def training_step(self, batch, batch_idx):
        img, target_seq = batch
        outputs = self.forward(img, target_seq)
        token_loss = self.criterion(outputs, target_seq)

        # Reconstruct image from predicted SVD tokens.
        reconstructed_img = self.reconstruct_from_svd_tokens(outputs)
        # x: (batch, 1, H, W) --> squeeze channel dimension: (batch, H, W)
        recon_loss = F.mse_loss(reconstructed_img, img.squeeze(1))

        # Combined loss.
        loss = token_loss + self.lambda_recon * recon_loss
        self.log("train_loss", loss)
        self.log("token_loss", token_loss)
        self.log("recon_loss", recon_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        img, target_seq = batch
        outputs = self.forward(img, target_seq, teacher_forcing_ratio=0.0)
        token_loss = self.criterion(outputs, target_seq)

        reconstructed_img = self.reconstruct_from_svd_tokens(outputs)
        recon_loss = F.mse_loss(reconstructed_img, img.squeeze(1))

        loss = token_loss + self.lambda_recon * recon_loss
        self.log("val_loss", loss)
        self.log("val_token_loss", token_loss)
        self.log("val_recon_loss", recon_loss)

        if batch_idx % 100 == 0:
            print(
                f"Epoch {self.current_epoch}, Step {batch_idx}, Token Loss: {token_loss.item()}, Recon Loss: {recon_loss.item()}")
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def generate_from_latent(self, latent, seq_len=None):
        """
        Generate SVD token sequence from a given latent vector.
        latent: tensor of shape (batch, latent_dim)
        seq_len: number of tokens to generate (default: self.k_components)
        """
        if seq_len is None:
            seq_len = self.k_components
        batch_size = latent.size(0)
        h0 = self.latent_to_hidden(latent).unsqueeze(0)
        c0 = self.latent_to_cell(latent).unsqueeze(0)
        outputs = []
        input_token = self.start_token.expand(batch_size, -1)  # initial token
        for t in range(seq_len):
            embed = self.decoder_embedding(input_token).unsqueeze(1)
            lstm_out, (h0, c0) = self.lstm(embed, (h0, c0))
            token_pred = self.output_layer(lstm_out.squeeze(1))
            outputs.append(token_pred.unsqueeze(1))
            input_token = token_pred  # autoregressive generation
        outputs = torch.cat(outputs, dim=1)  # shape: (batch, seq_len, token_dim)
        return outputs

    def reconstruct_from_svd_tokens(self, svd_tokens):
        """
        Reconstruct a grayscale image from a sequence of SVD tokens.
        svd_tokens: tensor of shape (batch, k_components, token_dim)
                    where token_dim = image_size + 1 + image_size
        Returns:
            Reconstructed image tensor of shape (batch, image_size, image_size)
        """
        batch, k, token_dim = svd_tokens.shape
        image_size = self.image_size
        # Split tokens into u, sigma, v.
        u = svd_tokens[..., :image_size]  # (batch, k, image_size)
        sigma = svd_tokens[..., image_size:image_size + 1]  # (batch, k, 1)
        v = svd_tokens[..., image_size + 1:]  # (batch, k, image_size)

        # Normalize u and v to enforce unit norm.
        u = u / (u.norm(dim=2, keepdim=True) + 1e-8)
        v = v / (v.norm(dim=2, keepdim=True) + 1e-8)
        # Ensure singular values are nonnegative.
        sigma = torch.relu(sigma)

        # Reconstruct the image as a sum of outer products.
        reconstructed = 0
        for i in range(k):
            u_i = u[:, i, :]  # (batch, image_size)
            v_i = v[:, i, :]  # (batch, image_size)
            sigma_i = sigma[:, i, :]  # (batch, 1)
            # Compute outer product: (batch, image_size, 1) * (batch, 1, image_size)
            u_i = u_i.unsqueeze(2)  # (batch, image_size, 1)
            v_i = v_i.unsqueeze(1)  # (batch, 1, image_size)
            outer = u_i * v_i  # (batch, image_size, image_size)
            reconstructed = reconstructed + sigma_i.unsqueeze(-1) * outer
        return reconstructed


# --------------------------
# Data Module for CIFAR-10
# --------------------------
class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir="data", batch_size=64, k_components=10):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.k_components = k_components

        # Convert to grayscale and to tensor.
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

    def setup(self, stage=None):
        self.cifar10_train = CIFAR10SVDDataset(
            train=True, transform=self.transform, k_components=self.k_components
        )
        self.cifar10_val = CIFAR10SVDDataset(
            train=False, transform=self.transform, k_components=self.k_components
        )

    def train_dataloader(self):
        return DataLoader(self.cifar10_train, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.cifar10_val, batch_size=self.batch_size, shuffle=False, num_workers=4)


# import matplotlib.pyplot as plt
# def display_samples(img, outputs):
#     # Display the original image.
#     img = img.squeeze(1).cpu().numpy()
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.imshow(img[0], cmap="gray")
#     plt.axis("off")
#     plt.title("Original Image")
#
#     # Display the predicted SVD components.
#     outputs = outputs.cpu().numpy()
#     plt.subplot(1, 2, 2)
#     plt.imshow(outputs[0].T, cmap="gray")
#     plt.axis("off")
#     plt.title("Predicted SVD Components")
#     plt.show()


# def generate_samples(model, dm, num_samples=5):
#     model.eval()
#     with torch.no_grad():
#         for i, (img, target_seq) in enumerate(dm.val_dataloader()):
#             if i >= num_samples:
#                 break
#             img = img.to(model.device)
#             target_seq = target_seq.to(model.device)
#             outputs = model(img, target_seq, teacher_forcing_ratio=0.0)
#             print(f"Sample {i + 1} - Image Shape: {img.shape}, Token Shape: {outputs.shape}")
#
#             display_samples(img, outputs)


# --------------------------
# Main: Training the Model
# --------------------------
if __name__ == "__main__":
    dm = CIFAR10DataModule(batch_size=64, k_components=10)
    model = SVDAutoRegressiveModel(latent_dim=128, k_components=10, image_size=32)
    trainer = pl.Trainer(max_epochs=10, accelerator="auto", fast_dev_run=False, enable_progress_bar=False)
    trainer.fit(model, dm)

    # Switch to evaluation mode.
    model.eval()

    # Sample random latent vectors (here, 4 images).
    latent = torch.randn(4, 128)  # adjust latent_dim if needed

    with torch.no_grad():
        generated_tokens = model.generate_from_latent(latent)
        generated_images = model.reconstruct_from_svd_tokens(generated_tokens)

    # Display generated images using matplotlib.
    import matplotlib.pyplot as plt

    generated_images = generated_images.cpu().numpy()

    fig, axs = plt.subplots(1, 4, figsize=(12, 3))
    for i in range(4):
        axs[i].imshow(generated_images[i], cmap='gray')
        axs[i].axis('off')
    plt.show()
