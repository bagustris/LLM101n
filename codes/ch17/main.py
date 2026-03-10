import torch
import torchvision
import torchvision.transforms as transforms
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "../data"
os.makedirs(DATA_DIR, exist_ok=True)

# torchvision downloads and caches CIFAR-10 automatically
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),   # → [-1, 1]
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
print(f"CIFAR-10: {len(train_dataset):,} train  |  {len(val_dataset):,} val")
print(f"Classes: {classes}")
print(f"Image shape: {train_dataset[0][0].shape}")


import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    Vector Quantisation (VQ) layer with straight-through estimator.

    In the forward pass, each latent vector z is replaced by the nearest
    codebook entry e_k. The gradient is passed straight through the
    argmin operation (zero gradient through the discretisation).

    Loss = ||z_e - sg[e]||² + β ||sg[z_e] - e]||²
    where sg = stop_gradient
    """

    def __init__(self, n_codes: int, code_dim: int, beta: float = 0.25):
        super().__init__()
        self.n_codes  = n_codes
        self.code_dim = code_dim
        self.beta     = beta
        # Codebook: n_codes embeddings of size code_dim
        self.codebook = nn.Embedding(n_codes, code_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / n_codes, 1.0 / n_codes)

    def forward(self, z: torch.Tensor):
        """
        z: (B, C, H, W)  — spatial latents from encoder
        Returns: (quantized, vq_loss, indices)
          quantized: (B, C, H, W) — codes replacing latents
          vq_loss:   scalar       — codebook + commitment loss
          indices:   (B, H*W)     — which codebook entry was selected
        """
        B, C, H, W = z.shape
        # Flatten spatial dims: (B, H*W, C)
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, C)

        # Distances to each codebook entry: ||z - e||² = ||z||² - 2z·e + ||e||²
        dist = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * z_flat @ self.codebook.weight.T
            + self.codebook.weight.pow(2).sum(dim=1)
        )
        indices = dist.argmin(dim=1)   # (B*H*W,)

        # Retrieve quantised vectors
        z_q = self.codebook(indices).view(B, H, W, C).permute(0, 3, 1, 2)

        # VQ loss: codebook loss + β * commitment loss
        # sg(x) = x.detach()
        vq_loss = (
            F.mse_loss(z_q.detach(), z) * self.beta      # commitment
            + F.mse_loss(z_q, z.detach())                # codebook
        )

        # Straight-through: gradients bypass quantisation
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
        return x + self.net(x)   # residual connection


class VQVAE(nn.Module):
    """
    VQVAE for 32×32 RGB images (CIFAR-10).
    Encoder: 32×32 → 8×8 spatial latents
    Decoder: 8×8 quantised latents → 32×32 images
    """

    def __init__(self, n_codes: int = 512, code_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),   # 32→16
            nn.ReLU(),
            nn.Conv2d(64, code_dim, 4, stride=2, padding=1),  # 16→8
            nn.ReLU(),
            ResBlock(code_dim),
            ResBlock(code_dim),
        )
        self.vq        = VectorQuantizer(n_codes, code_dim)
        self.decoder   = nn.Sequential(
            ResBlock(code_dim),
            ResBlock(code_dim),
            nn.ConvTranspose2d(code_dim, 64, 4, stride=2, padding=1),  # 8→16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),         # 16→32
            nn.Tanh(),   # output in [-1, 1]
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        z_q, vq_loss, indices = self.vq(z)
        return z_q, indices, vq_loss

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_q)

    def forward(self, x: torch.Tensor):
        z_q, indices, vq_loss = self.encode(x)
        x_hat = self.decode(z_q)
        return x_hat, vq_loss, indices


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = VQVAE(n_codes=512, code_dim=64).to(device)
print(f"VQVAE parameters: {sum(p.numel() for p in model.parameters()):,}")

optim = torch.optim.Adam(model.parameters(), lr=2e-4)

EPOCHS      = 5
LOG_EVERY   = 200
total_steps = 0

for epoch in range(EPOCHS):
    model.train()
    epoch_recon = 0.0
    epoch_vq    = 0.0

    for step, (imgs, _labels) in enumerate(train_loader):
        imgs = imgs.to(device)

        x_hat, vq_loss, _ = model(imgs)

        # Reconstruction loss (MSE in pixel space)
        recon_loss = F.mse_loss(x_hat, imgs)
        loss       = recon_loss + vq_loss

        optim.zero_grad()
        loss.backward()
        optim.step()

        epoch_recon += recon_loss.item()
        epoch_vq    += vq_loss.item()
        total_steps += 1

        if total_steps % LOG_EVERY == 0:
            print(f"Epoch {epoch}  step {step}  "
                  f"recon={recon_loss.item():.4f}  vq={vq_loss.item():.4f}")

    n = len(train_loader)
    print(f"Epoch {epoch} complete — "
          f"avg recon: {epoch_recon/n:.4f}  avg vq: {epoch_vq/n:.4f}")

torch.save(model.state_dict(), "../data/vqvae_cifar10.pt")
print("VQVAE checkpoint saved → data/vqvae_cifar10.pt")


model.eval()
imgs_val, labels_val = next(iter(val_loader))
imgs_val = imgs_val[:8].to(device)

with torch.no_grad():
    recons, _, indices = model(imgs_val)

# De-normalise from [-1, 1] → [0, 1]
def denorm(t):
    return (t * 0.5 + 0.5).clamp(0, 1)

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

# Show image tokens
print(f"\nImage token indices for first image: {indices[0].tolist()[:16]} …")
print(f"Codebook size: {model.vq.n_codes}  |  Tokens per image: {indices.shape[1]}")


print("""
Multimodal Architecture: Image + Text → Language Model
═══════════════════════════════════════════════════════

Step 1: Image Tokenisation (VQVAE encoder + codebook lookup)
  image (3×32×32)
    ↓ CNN encoder
  spatial latents (64×8×8)
    ↓ VQ (nearest codebook entry)
  image tokens: [42, 311, 7, 99, …]  (64 tokens for 8×8 grid)

Step 2: Combined Vocabulary
  - Text tokens:  indices 0 … vocab_size-1
  - Image tokens: indices vocab_size … vocab_size + n_codes - 1
  - Special tokens: [IMG_START], [IMG_END], [TXT_START], [EOS]

Step 3: Interleaved Sequence
  [TXT_START] "A photo of a cat" [IMG_START] t1 t2 … t64 [IMG_END] [EOS]

Step 4: Train a GPT on combined sequences
  - Same causal LM objective: predict next token
  - Works for text-to-image, image-to-text, or interleaved

This is essentially how LLaVA, Gemini, and Chameleon work.
""")


print("""
Diffusion Transformer (DiT) — Key Components
══════════════════════════════════════════════

1. Forward diffusion:  q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t)I)
   Gradually adds Gaussian noise over T timesteps.

2. Reverse process (denoising):
   p_θ(x_{t-1} | x_t, c) — transformer predicts noise ε_θ(x_t, t, c)

3. DiT architecture:
   Patch embed → N × (Attention + AdaLayerNorm + FFN) → Linear head
   
   AdaLayerNorm conditions on timestep t and class label c:
     γ, β = MLP(embed(t) + embed(c))
     output = γ * LayerNorm(x) + β

4. Training objective (simple ε-prediction):
   L = E[||ε - ε_θ(√ᾱ_t x_0 + √(1-ᾱ_t) ε, t, c)||²]

5. Sampling (DDPM / DDIM):
   Start from x_T ~ N(0, I), iteratively denoise to x_0.
   DDIM can generate in 20-50 steps instead of 1000.

Key insight: by treating image patches as tokens, DiT gets all the
scaling benefits of transformers (depth, width, data parallelism)
that made GPT-3 and GPT-4 work for language.
""")


# Analyse codebook usage — a well-trained VQVAE uses most codes
model.eval()
all_indices = []

with torch.no_grad():
    for imgs, _ in val_loader:
        _, _, indices = model(imgs.to(device))
        all_indices.append(indices.cpu())

all_indices = torch.cat(all_indices, dim=0).view(-1)
n_codes     = model.vq.n_codes
usage       = torch.bincount(all_indices, minlength=n_codes).float()
usage_frac  = (usage > 0).float().mean().item()

print(f"Codebook utilisation: {usage_frac:.1%} of {n_codes} codes used")

# Plot code usage frequency
plt.figure(figsize=(10, 4))
plt.bar(range(n_codes), usage.sort(descending=True).values.numpy(), width=1.0)
plt.xlabel("Code index (sorted by frequency)")
plt.ylabel("Usage count")
plt.title(f"VQVAE Codebook Usage (CIFAR-10 validation) — {usage_frac:.0%} active")
plt.tight_layout()
plt.savefig("../data/ch17_codebook_usage.png", dpi=100)
print("Saved → data/ch17_codebook_usage.png")
