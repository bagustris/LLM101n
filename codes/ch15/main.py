import torch
import torch.nn as nn
import torch.nn.functional as F
import os

DATA_DIR = "../data"


class RewardModel(nn.Module):
    """
    Standalone reward model built on a simple transformer.
    Outputs a scalar reward score for each input sequence.
    """

    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, max_len: int = 64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4*d_model,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.reward_head = nn.Linear(d_model, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Returns reward scalars of shape (B,)."""
        B, T = input_ids.shape
        device = input_ids.device
        x = self.emb(input_ids) + self.pos_emb(
            torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        )
        x = self.transformer(x)
        reward = self.reward_head(x[:, -1, :]).squeeze(-1)
        return reward


def reward_model_loss(model: nn.Module, chosen: torch.Tensor,
                      rejected: torch.Tensor) -> torch.Tensor:
    """
    Bradley-Terry pairwise ranking loss.
    Loss = -mean(log sigmoid(r_w - r_l))
    """
    reward_chosen   = model(chosen)
    reward_rejected = model(rejected)
    logits = reward_chosen - reward_rejected
    loss   = -F.logsigmoid(logits).mean()
    return loss


def dpo_loss(
    policy_log_probs_chosen:   torch.Tensor,
    policy_log_probs_rejected: torch.Tensor,
    ref_log_probs_chosen:      torch.Tensor,
    ref_log_probs_rejected:    torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    Direct Preference Optimisation loss (Rafailov et al., 2023).
    """
    log_ratio_chosen   = policy_log_probs_chosen   - ref_log_probs_chosen
    log_ratio_rejected = policy_log_probs_rejected - ref_log_probs_rejected
    reward_diff = beta * (log_ratio_chosen - log_ratio_rejected)
    loss = -F.logsigmoid(reward_diff).mean()
    return loss


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(DATA_DIR, exist_ok=True)

    try:
        from datasets import load_dataset
        hh_dataset = load_dataset("Anthropic/hh-rlhf", split="train[:2000]")
        print(f"Loaded {len(hh_dataset)} preference pairs")
    except Exception as e:
        print(f"Could not load hh-rlhf: {e}")
        hh_dataset = [
            {"chosen":   "The sky is blue because of Rayleigh scattering.",
             "rejected": "The sky is blue because God painted it."},
        ]
        print("Using synthetic preference data.")

    torch.manual_seed(42)
    B = 8
    r_chosen   = torch.randn(B) + 1.0
    r_rejected = torch.randn(B) - 1.0
    logits = r_chosen - r_rejected
    loss_rm = -nn.functional.logsigmoid(logits).mean()
    acc_rm  = (r_chosen > r_rejected).float().mean()
    print(f"Reward model loss: {loss_rm.item():.4f}")
    print(f"Ranking accuracy : {acc_rm.item():.2%}")

    B = 16
    policy_chosen   = torch.randn(B) - 0.5
    policy_rejected = torch.randn(B) - 1.0
    ref_chosen      = torch.randn(B) - 1.0
    ref_rejected    = torch.randn(B) - 1.0
    loss_dpo = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
    print(f"DPO loss: {loss_dpo.item():.4f}")

    beta_values = np.linspace(0.001, 1.0, 100)
    simulated_reward = 5.0 / (1.0 + beta_values * 10)
    simulated_kl     = 1.0 / (beta_values + 0.1) - 0.9

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    ax1.plot(beta_values, simulated_reward, color="steelblue", label="Reward")
    ax2.plot(beta_values, simulated_kl.clip(0), color="coral", label="KL divergence")
    ax1.set_xlabel("Beta (KL penalty coefficient)")
    ax1.set_ylabel("Reward", color="steelblue")
    ax2.set_ylabel("KL(pi || pi_ref)", color="coral")
    ax1.set_title("Trade-off: Reward vs KL divergence")
    plt.tight_layout()
    plt.savefig("../data/ch15_rlhf_tradeoff.png", dpi=100)
    print("Saved -> data/ch15_rlhf_tradeoff.png")
