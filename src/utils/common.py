import torch
import os

STR2PTDTYPE = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}

def _check_path_exists(path):
    if os.path.exists(path):
        return True
    if os.path.exists(os.path.join(os.getcwd(), path)):
        return True
    return False