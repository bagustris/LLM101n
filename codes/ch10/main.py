import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler


def main():
    rank       = int(os.environ.get("RANK",       0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    if rank == 0:
        print(f"World size: {world_size}  |  Backend: {backend}")

    model = nn.Sequential(
        nn.Linear(512, 2048), nn.GELU(),
        nn.Linear(2048, 2048), nn.GELU(),
        nn.Linear(2048, 512),
    ).to(device)

    model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    N = 10_000
    X = torch.randn(N, 512)
    Y = torch.randn(N, 512)
    dataset = TensorDataset(X, Y)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                 shuffle=True, drop_last=True)
    loader  = DataLoader(dataset, batch_size=64, sampler=sampler,
                         num_workers=2, pin_memory=True)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(3):
        sampler.set_epoch(epoch)
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out    = model(xb)
            loss   = nn.functional.mse_loss(out, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()

        if rank == 0:
            print(f"Epoch {epoch}  loss: {total_loss / len(loader):.4f}")

    if rank == 0:
        torch.save(model.module.state_dict(), "../data/ddp_checkpoint.pt")
        print("Checkpoint saved -> data/ddp_checkpoint.pt")

    dist.destroy_process_group()


def simulate_allreduce(tensor_size: int, world_size: int) -> str:
    """
    Ring all-reduce sends 2*(world_size-1)/world_size * data per GPU.
    """
    bytes_per_elem = 4
    total_bytes    = tensor_size * bytes_per_elem
    traffic_bytes  = 2 * (world_size - 1) / world_size * total_bytes
    return (f"  Tensor: {total_bytes/1e6:.1f} MB  "
            f"| Per-GPU traffic: {traffic_bytes/1e6:.1f} MB  "
            f"| At 600 GB/s NVLink: {traffic_bytes/600e9*1e3:.2f} ms")


if __name__ == "__main__":
    print("DDP script defined. Launch with: torchrun --nproc_per_node=2 ch10/main.py")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = nn.Sequential(
        nn.Linear(256, 1024), nn.GELU(),
        nn.Linear(1024, 256),
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)

    MICRO_BATCH_SIZE = 32
    ACCUM_STEPS      = 8
    print(f"Effective batch size: {MICRO_BATCH_SIZE * ACCUM_STEPS}")

    for step in range(20):
        accum_loss = 0.0
        for micro_step in range(ACCUM_STEPS):
            xb = torch.randn(MICRO_BATCH_SIZE, 256, device=device)
            yb = torch.randn(MICRO_BATCH_SIZE, 256, device=device)
            out  = model(xb)
            loss = nn.functional.mse_loss(out, yb) / ACCUM_STEPS
            loss.backward()
            accum_loss += loss.item()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        optim.zero_grad()
        if step % 5 == 0:
            print(f"Step {step}  loss: {accum_loss:.4f}")

    param_counts = {
        "GPT-2 Small (117M)": 117_000_000,
        "GPT-2 Large (774M)": 774_000_000,
        "LLaMA-7B":           7_000_000_000,
    }

    for name, params in param_counts.items():
        print(f"\n{name} ({params/1e6:.0f}M params), world_size=8:")
        print(simulate_allreduce(params, world_size=8))

    import json
    ds_config = {
        "train_batch_size": 256,
        "gradient_accumulation_steps": 8,
        "fp16": {"enabled": True},
        "zero_optimization": {"stage": 2}
    }
    os.makedirs("../data", exist_ok=True)
    with open("../data/ds_config.json", "w") as f:
        json.dump(ds_config, f, indent=2)
    print("DeepSpeed config saved -> data/ds_config.json")
