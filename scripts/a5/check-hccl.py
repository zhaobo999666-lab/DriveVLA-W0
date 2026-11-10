"""Verify one HCCL collective across all sixteen NPU ranks."""

import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    if int(os.environ["WORLD_SIZE"]) != 16:
        raise RuntimeError("Expected two nodes with eight NPU ranks each")
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    try:
        value = torch.tensor([rank + 1], dtype=torch.int32, device=f"npu:{local_rank}")
        dist.all_reduce(value)
        if value.item() != 136:
            raise RuntimeError(f"Rank {rank}: incorrect sum {value.item()}")
        print(f"HCCL OK rank={rank} local_rank={local_rank} sum=136", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
