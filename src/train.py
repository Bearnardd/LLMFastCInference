import os
import math
import os
import time
from contextlib import nullcontext
from datetime import datetime
from functools import partial
from pathlib import Path
import json

import torch
# from model import Transformer, ModelArgs
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.dist import init_ddp
from utils.common import STR2PTDTYPE
from utils.config import TrainingConfig
from datasets import Task


from models import GenericTransformerModel


if __name__ == "__main__":

    # cfg = TrainingConfig.from_json("config.json")
    # ddp = int(os.environ.get("RANK", -1)) != -1
    # if ddp:
    #     ddp_rank, ddp_local_rank, ddp_world_size = init_ddp(cfg)
    #     master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    #     seed_offset = ddp_rank  # each process gets a different seed
    # else:
    #     # if not ddp, we are running on a single gpu, and one process
    #     master_process = True
    #     seed_offset = 0
    #     ddp_world_size = 1

    # tokens_per_iter = cfg.gradient_accumulation_steps * ddp_world_size * cfg.batch_size * cfg.max_seq_len

    # if master_process:
    #     print(f"tokens per iteration will be: {tokens_per_iter:,}")
    #     print(f"breaks down as: {cfg.gradient_accumulation_steps} grad accum steps * {ddp_world_size} processes * {cfg.batch_size} batch size * {cfg.max_seq_len} max seq len")

    # if master_process:
    #     os.makedirs(cfg.output_dir, exist_ok=True)

    # torch.manual_seed(1337 + seed_offset)
    # torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    # torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    # device_type = "cuda" if "cuda" in cfg.device else "cpu"  # for later use in torch.autocast
    # # note: float16 data type will automatically use a GradScaler
    # ctx = (
    #     nullcontext()
    #     if device_type == "cpu"
    #     else torch.amp.autocast(device_type=device_type, dtype=STR2PTDTYPE[cfg.dtype])
    # )

    model = GenericTransformerModel.from_json("config.json")
    import pdb;pdb.set_trace()

    # scaler = torch.cuda.amp.GradScaler(enabled=(cfg.dtype == "float16"))

# TODO: add resuming training from a checkpoint
# elif init_from == "resume":
#     print(f"Resuming training from {out_dir}")
#     # resume training from a checkpoint.
#     ckpt_path = os.path.join(out_dir, "ckpt.pt")
#     checkpoint = torch.load(ckpt_path, map_location=device)
#     checkpoint_model_args = checkpoint["model_args"]
#     # force these config attributes to be equal otherwise we can't even resume training
#     # the rest of the attributes (e.g. dropout) can stay as desired from command line
#     for k in ["dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "multiple_of", "max_seq_len"]:
#         model_args[k] = checkpoint_model_args[k]
#     # create the model
#     gptconf = ModelArgs(**model_args)
#     model = Transformer(gptconf)
#     state_dict = checkpoint["model"]
#     # fix the keys of the state dictionary :(
#     # honestly no idea how checkpoints sometimes get this prefix, have to debug more
#     unwanted_prefix = "_orig_mod."
#     for k, v in list(state_dict.items()):
#         if k.startswith(unwanted_prefix):
#             state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
#     model.load_state_dict(state_dict)
#     iter_num = checkpoint["iter_num"]
#     best_val_loss = checkpoint["best_val_loss"]
# model.to(device)