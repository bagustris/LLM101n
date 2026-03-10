import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "../data"


class VectorQuantizer(nn.Module):
    """Vector Quantisation layer with straight-through estimator."""

    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim  = embedding_dim
        self.beta           = beta
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z: torch.Tensor):
        """
        z: (B, C, H, W)
        Returns: (quantized, vq_loss, indices)
        """
        B, C, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, C)

        dist = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * z_flat @ self.codebook.weight.T
            + self.codebook.weight.pow(2).sum(dim=1)
        )
        indices = dist.argmin(dim=1)

        z_q = self.codebook(indices).view(B, H, W, C).permute(0, 3, 1, 2)

        vq_loss = (
            F.mse_loss(z_q.detach(), z) * self.beta
            + F.mse_loss(z_q, z.detach())
        )

        z_q_st = z + (z_q - z).detach()

        return z_q_st, vq_loss, indices.view(B, H * W)


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class VQVAE(nn.Module):
    """VQVAE for RGB images."""

    def __init__(self, in_channels: int = 3, hidden_dim: int = 64,
                 num_embeddings: int = 512, embedding_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, embedding_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            ResBlock(embedding_dim),
            ResBlock(embedding_dim),
        )
        self.vq      = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = nn.Sequential(
            ResBlock(embedding_dim),
            ResBlock(embedding_dim),
            nn.ConvTranspose2d(embedding_dim, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, in_channels, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor):
        z = self.encoder(x)
        z_q, vq_loss, indices = self.vq(z)
        return z_q, indices, vq_loss

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_q)

    def forward(self, x: torch.Tensor):
        z_q, indices, vq_loss = self.encode(x)
        x_hat = self.decode(z_q)
        return x_hat, vq_loss


def denorm(t: torch.Tensor) -> torch.Tensor:
    """De-normalise from [-1, 1] to [0, 1]."""
    return (t * 0.5 + 0.5).clamp(0, 1)


if __name__ == "__main__":
    import torchvision
    import torchvision.transforms as transforms

    os.makedirs(DATA_DIR, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=transform
    )
    val_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=2
    )

    classes = train_dataset.classes
    print(f"CIFAR-10: {len(train_dataset):,} train | {len(val_dataset):,} val")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = VQVAE(in_channels=3, hidden_dim=64, num_embeddings=512, embedding_dim=64).to(device)
    print(f"VQVAE parameters: {sum(p.numel() for p in model.parameters()):,}")

    optim = torch.optim.Adam(model.parameters(), lr=2e-4)

    EPOCHS    = 5
    LOG_EVERY = 200
    total_steps = 0

    for epoch in range(EPOCHS):
        model.train()
        epoch_recon = 0.0
        epoch_vq    = 0.0

        for step, (imgs, _labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            x_hat, vq_loss = model(imgs)
            recon_loss = F.mse_loss(x_hat, imgs)
            loss       = recon_loss + vq_loss
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_recon += recon_loss.item()
            epoch_vq    += vq_loss.item()
            total_steps += 1
            if total_steps % LOG_EVERY == 0:
                print(f"Epoch {epoch} step {step} recon={recon_loss.item():.4f} vq={vq_loss.item():.4f}")

        n = len(train_loader)
        print(f"Epoch {epoch} — avg recon: {epoch_recon/n:.4f} avg vq: {epoch_vq/n:.4f}")

    torch.save(model.state_dict(), "../data/vqvae_cifar10.pt")
    print("VQVAE checkpoint saved → data/vqvae_cifar10.pt")

    model.eval()
    imgs_val, labels_val = next(iter(val_loader))
    imgs_val = imgs_val[:8].to(device)

    with torch.no_grad():
        recons, _ = model(imgs_val)

    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i in range(8):
        axes[0, i].imshow(denorm(imgs_val[i]).permute(1, 2, 0).cpu())
        axes[0, i].set_title(classes[labels_val[i]])
        axes[0, i].axis("off")
        axes[1, i].imshow(denorm(recons[i]).permute(1, 2, 0).cpu())
        axes[1, i].set_title("Recon")
        axes[1, i].axis("off")

    plt.suptitle("VQVAE: Original (top) vs Reconstruction (bottom)")
    plt.tight_layout()
    plt.savefig("../data/ch17_vqvae_recons.png", dpi=100)
    print("Saved → data/ch17_vqvae_recons.png")
