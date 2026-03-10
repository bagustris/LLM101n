from datasets import load_dataset
import torch
import torch.nn as nn
import os

DATA_DIR = "../data"
os.makedirs(DATA_DIR, exist_ok=True)

# Anthropic/hh-rlhf: human preference data with chosen/rejected responses
print("Loading preference dataset …")
try:
    hh_dataset = load_dataset("Anthropic/hh-rlhf", split="train[:2000]")
    print(f"Loaded {len(hh_dataset)} preference pairs")
    print(f"Columns: {hh_dataset.column_names}")
    print(f"\nSample chosen  : {hh_dataset[0]['chosen'][:200]}")
    print(f"Sample rejected: {hh_dataset[0]['rejected'][:200]}")
except Exception as e:
    print(f"Could not load hh-rlhf: {e}")
    # Create synthetic preference data for demonstration
    hh_dataset = [
        {"chosen":   "The sky is blue because of Rayleigh scattering of sunlight.",
         "rejected": "The sky is blue because God painted it."},
        {"chosen":   "Water boils at 100°C at standard atmospheric pressure.",
         "rejected": "Water boils when it feels like it."},
    ]
    print("Using synthetic preference data for demonstration.")


from transformers import AutoModel, AutoTokenizer

class RewardModel(nn.Module):
    """
    Reward model built on top of a pretrained encoder.
    Outputs a scalar reward score for a given (prompt, response) string.
    """

    def __init__(self, base_model_name: str = "gpt2"):
        super().__init__()
        from transformers import AutoModelForSequenceClassification
        # Use a sequence classification head that outputs a single scalar
        self.model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=1,
        )

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Returns reward scalars of shape (B,)."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits.squeeze(-1)


# Bradley-Terry reward model training objective:
# maximize log σ(r_chosen - r_rejected)
def reward_model_loss(
    reward_chosen:   torch.Tensor,   # (B,) reward scores for chosen responses
    reward_rejected: torch.Tensor,   # (B,) reward scores for rejected responses
) -> torch.Tensor:
    """
    Bradley-Terry pairwise ranking loss.
    We want reward_chosen > reward_rejected.
    Loss = -mean(log sigmoid(r_w - r_l))
    """
    logits = reward_chosen - reward_rejected
    loss   = -nn.functional.logsigmoid(logits).mean()
    return loss


# Demo: random rewards
torch.manual_seed(42)
B = 8
r_chosen   = torch.randn(B) + 1.0   # biased positive
r_rejected = torch.randn(B) - 1.0   # biased negative
loss_rm    = reward_model_loss(r_chosen, r_rejected)
acc_rm     = (r_chosen > r_rejected).float().mean()
print(f"Reward model loss: {loss_rm.item():.4f}")
print(f"Ranking accuracy : {acc_rm.item():.2%}")


print("""
PPO for LLMs — Training Loop (Conceptual)
──────────────────────────────────────────

The PPO objective maximises:
  L_PPO = E[ min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t) ]
         − β * KL(π_θ || π_ref)

where:
  r_t    = π_θ(a_t|s_t) / π_ref(a_t|s_t)  (probability ratio)
  A_t    = reward_model(response) - baseline
  β      = KL penalty coefficient (typically 0.01–0.1)
  π_ref  = frozen SFT model (prevents reward hacking)

Training loop (one iteration):
  1. ROLLOUT:   sample responses from current policy π_θ
  2. SCORE:     get reward r = reward_model(prompt, response)
  3. KL PENALTY: compute KL(π_θ || π_ref) per token
  4. ADVANTAGE: compute A = r - V(s) where V is a value head
  5. PPO UPDATE: update π_θ using clipped surrogate + value loss

Libraries implementing full PPO for LLMs:
  • TRL (HuggingFace): trl.PPOTrainer
  • DeepSpeed-Chat: deepspeed.runtime.rlhf
  • OpenRLHF: github.com/OpenLLMAI/OpenRLHF
""")


def dpo_loss(
    policy_log_probs_chosen:   torch.Tensor,   # (B,) log π_θ(y_w|x)
    policy_log_probs_rejected: torch.Tensor,   # (B,) log π_θ(y_l|x)
    ref_log_probs_chosen:      torch.Tensor,   # (B,) log π_ref(y_w|x)
    ref_log_probs_rejected:    torch.Tensor,   # (B,) log π_ref(y_l|x)
    beta: float = 0.1,
) -> torch.Tensor:
    """
    Direct Preference Optimisation loss (Rafailov et al., 2023).

    DPO eliminates the reward model by showing that the optimal policy
    under the RLHF objective satisfies:

      r*(x,y) = β * log[π*(y|x) / π_ref(y|x)] + β * log Z(x)

    Substituting into the Bradley-Terry preference model and simplifying:

      L_DPO = -E[ log σ(β * (log π(y_w|x) - log π_ref(y_w|x))
                         - β * (log π(y_l|x) - log π_ref(y_l|x))) ]
    """
    log_ratio_chosen   = policy_log_probs_chosen   - ref_log_probs_chosen
    log_ratio_rejected = policy_log_probs_rejected - ref_log_probs_rejected

    # The implicit reward difference
    reward_diff = beta * (log_ratio_chosen - log_ratio_rejected)

    # DPO loss: negative log sigmoid of reward difference
    loss = -nn.functional.logsigmoid(reward_diff).mean()
    return loss


# Demo: policy that correctly prefers chosen over rejected
torch.manual_seed(42)
B = 16
# Simulate log-probs: chosen has higher probability under policy than ref
policy_chosen   = torch.randn(B) - 0.5    # policy mildly prefers chosen
policy_rejected = torch.randn(B) - 1.0
ref_chosen      = torch.randn(B) - 1.0    # reference assigns roughly equal probs
ref_rejected    = torch.randn(B) - 1.0

loss_dpo = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
print(f"DPO loss: {loss_dpo.item():.4f}")


print("""
DPO Training with TRL (HuggingFace)
────────────────────────────────────
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig

model     = AutoModelForCausalLM.from_pretrained("gpt2")
ref_model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Apply LoRA to keep training cheap
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn"])
model = get_peft_model(model, lora_config)

# Dataset must have columns: prompt, chosen, rejected
from datasets import load_dataset
dataset = load_dataset("Anthropic/hh-rlhf", split="train[:5000]")

dpo_config = DPOConfig(
    beta            = 0.1,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4,
    learning_rate   = 5e-5,
    num_train_epochs= 1,
    output_dir      = "../data/dpo_output",
)

trainer = DPOTrainer(
    model=model, ref_model=ref_model,
    args=dpo_config, tokenizer=tokenizer,
    train_dataset=dataset,
)
trainer.train()
""")


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Demonstrate the trade-off between maximising reward and staying close to SFT
beta_values = np.linspace(0.001, 1.0, 100)

# Simulated: as beta → 0, policy drifts more but gets higher reward
# as beta → ∞, policy stays close to reference but reward is unconstrained
simulated_reward = 5.0 / (1.0 + beta_values * 10)   # decreasing reward
simulated_kl     = 1.0 / (beta_values + 0.1) - 0.9   # increasing KL

fig, ax1 = plt.subplots(figsize=(8, 4))
ax2 = ax1.twinx()
ax1.plot(beta_values, simulated_reward, color="steelblue", label="Reward")
ax2.plot(beta_values, simulated_kl.clip(0), color="coral", label="KL divergence")
ax1.set_xlabel("Beta (KL penalty coefficient)")
ax1.set_ylabel("Reward (higher is better)", color="steelblue")
ax2.set_ylabel("KL(π || π_ref) (lower is better)", color="coral")
ax1.set_title("Trade-off: Reward vs KL divergence under RLHF")
plt.tight_layout()
plt.savefig("../data/ch15_rlhf_tradeoff.png", dpi=100)
print("Saved → data/ch15_rlhf_tradeoff.png")
