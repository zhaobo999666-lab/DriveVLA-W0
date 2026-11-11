"""Verify one HCCL collective across all NPU ranks (8 = single node, 16 = two)."""

import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size not in (8, 16):
        raise RuntimeError(f"Expected 8 ranks (one node) or 16 ranks (two nodes), got {world_size}")
    expected_sum = world_size * (world_size + 1) // 2
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    try:
        value = torch.tensor([rank + 1], dtype=torch.int32, device=f"npu:{local_rank}")
        dist.all_reduce(value)
        if value.item() != expected_sum:
            raise RuntimeError(f"Rank {rank}: incorrect sum {value.item()} != {expected_sum}")
        print(f"HCCL OK rank={rank} local_rank={local_rank} sum={expected_sum}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
