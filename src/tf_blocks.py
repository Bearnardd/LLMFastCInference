
from modules.mistral import MistralAttention, MistralFeedForward
from modules.common import RMSNorm, NORM_STR_TO_NORM

import torch.nn as nn

class MistralBlock(nn.Module):
    def __init__(self,
                n_heads: int,
                n_kv_heads: int,
                dim: int,
                hidden_dim: int,
                head_dim: int,
                max_batch_size: int,
                sliding_window: int,
                norm_eps: float,
                dropout: float,
                multiple_of: int,
                rotary_emb: str) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.n_kv_heads = n_kv_heads
        self.dropout = dropout
        self.multiple_of = multiple_of
        self.rotary_emb = rotary_emb
        self.norm_eps = norm_eps
        self.attention = MistralAttention(n_heads=n_heads,
                                          n_kv_heads=n_kv_heads,
                                          dim=dim, head_dim=head_dim,
                                          sliding_window=sliding_window,
                                          max_batch_size=max_batch_size)
        self.ffn = MistralFeedForward(dim=dim, hidden_dim=hidden_dim)
        self.ffn_norm = RMSNorm(dim, norm_eps=norm_eps)


class LLaMABlock:
    pass



MODEL_TYPE_TO_BLOCK_TYPE = {
    "mistral": MistralBlock,
    "llama": LLaMABlock,
}