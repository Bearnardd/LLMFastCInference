import os
import torch
from torch.distributed import init_process_group

def get_ddp_ranks():
    return int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])

def init_ddp(cfg):
    init_process_group(backend="nccl")
    ddp_rank, ddp_local_rank, ddp_world_size = get_ddp_ranks()
    assert cfg.gradient_accumulation_steps % ddp_world_size == 0, "gradient_accumulation_steps must be divisible by world_size"
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    cfg.device = device
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    cfg.gradient_accumulation_steps //= ddp_world_size
    return ddp_rank, ddp_local_rank, ddp_world_size, device