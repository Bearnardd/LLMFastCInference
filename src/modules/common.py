import torch
import torch.nn as nn

from typing import Tuple, List
import inspect

def get_init_args(cls) -> List[str]:
    return inspect.getfullargspec(cls.__init__).args[1:]


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, norm_eps: float):
        super().__init__()
        self.eps = norm_eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.norm_eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

NORM_STR_TO_NORM_NAME_AND_ARGS = {
    "rmsnorm": [RMSNorm, ["dim", "norm_eps"]],
}

def repeat_kv(keys: torch.Tensor, values: torch.Tensor, repeats: int):
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=2)
    values = torch.repeat_interleave(values, repeats=repeats, dim=2)
    return keys, values


def _reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    freqs_cis: complex - (seq_len, head_dim / 2)
    x: complex - (bsz, seq_len, head_dim / 2)
    """
    ndim = x.ndim
    assert 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), (
        freqs_cis.shape,
        (x.shape[1], x.shape[-1]),
    )
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

# TODO: is it gptj or gptneox>
def apply_gptj_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = _reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)



NORM_STR_TO_NORM_NAME_AND_ARGS = {
    "rmsnorm": [RMSNorm, get_init_args(RMSNorm)],
}

NORM_STR_TO_NORM = {
    "rmsnorm": RMSNorm
}