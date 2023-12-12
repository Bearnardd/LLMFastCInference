from config import TrainingConfig
import os
import math
import os
import time
from contextlib import nullcontext
from datetime import datetime
from functools import partial

import torch
# from model import Transformer, ModelArgs
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.dist import init_ddp
from utils.misc import STR2PTDTYPE
from datasets import Task





if __name__ == "__main__":
    cfg = TrainingConfig("config.yaml")
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        ddp_rank, ddp_local_rank, ddp_world_size, device = init_ddp(cfg)
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
    tokens_per_iter = cfg.gradient_accumulation_steps * ddp_world_size * cfg.batch_size * cfg.max_seq_len

    if master_process:
        print(f"tokens per iteration will be: {tokens_per_iter:,}")
        print(f"breaks down as: {cfg.gradient_accumulation_steps} grad accum steps * {ddp_world_size} processes * {cfg.batch_size} batch size * {cfg.max_seq_len} max seq len")

    if master_process:
        os.makedirs(cfg.output_dir, exist_ok=True)

    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = "cuda" if "cuda" in cfg.device else "cpu"  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=STR2PTDTYPE[cfg.dtype])
    )