

import torch
from typing import Tuple
from abc import ABC, abstractmethod
import torch




class RopePositionalEmbeddings(ABC):
    @abstractmethod
    def precompute_freqs_cis(self, num_entities: int, emb_dim: int) -> torch.Tensor:
        """
        Precompute the frequencies of the sinusoidal positional embeddings for RotatE.

        Args:
        - num_entities (int): Number of entities.
        - emb_dim (int): Embedding dimension.

        Returns:
        - freqs_cis (torch.Tensor): Tensor containing the precomputed frequencies of sinusoidal positional embeddings.
        """
        pass

    @abstractmethod
    def apply_rotary_emb(self, query: torch.Tensor, key: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
        """
        Apply the sinusoidal positional embeddings to the query and key vectors.

        Args:
        - query (torch.Tensor): Query tensor.
        - key (torch.Tensor): Key tensor.
        - pos_emb (torch.Tensor): Sinusoidal positional embeddings.

        Returns:
        - rotated_query (torch.Tensor): Query tensor with applied positional embeddings.
        - rotated_key (torch.Tensor): Key tensor with applied positional embeddings.
        """
        pass


class GPTJRopePositionalEmbeddings(RopePositionalEmbeddings):
    @staticmethod
    def precompute_freqs_cis(dim: int, end: int, theta: float) -> torch.Tensor:
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device)  # type: ignore
        freqs = torch.outer(t, freqs).float()  # type: ignore
        return torch.polar(torch.ones_like(freqs), freqs)  # complex64

    @staticmethod
    def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis = freqs_cis[:, None, :]
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
        return xq_out.type_as(xq), xk_out.type_as(xk)



class GPTNeoXRopePositionalEmbeddings(RopePositionalEmbeddings):
    @staticmethod
    def precompute_freqs_cis(dim: int, end: int, theta: float) -> torch.Tensor:
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device)  # type: ignore
        freqs = torch.outer(t, freqs).float()  # type: ignore
        return torch.polar(torch.ones_like(freqs), freqs)  # complex64
    @staticmethod
    def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis = freqs_cis[:, None, :]
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
        return xq_out.type_as(xq), xk_out.type_as(xk)


MODEL_TO_POS_EMB_TYPE = {
    "generic": GPTJRopePositionalEmbeddings,
    "mistral": GPTNeoXRopePositionalEmbeddings,
}