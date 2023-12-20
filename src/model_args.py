from dataclasses import dataclass, fields
from simple_parsing.helpers import Serializable
from tf_blocks import MistralBlock



# TODO: change that, norm and output_laayers should be part of MistralArgs etc.?
@dataclass
class ModelArgs(Serializable):
    type: str
    n_layers: int
    vocab_size: int
    output_layers: str


BASE_MODEL_ARG_NAMES = [field.name for field in fields(ModelArgs)]

@dataclass
class MistralArgs(ModelArgs):
    dropout: float 
    n_heads: int 
    n_kv_heads: int
    dim: int
    head_dim: int
    hidden_dim: int
    multiple_of: int
    rotary_emb: str
    max_batch_size: int
    sliding_window: int
    norm_eps: float



MODEL_TYPE_TO_MODEL_ARGS = {
    "generic": ModelArgs,
    "mistral": MistralArgs,
}


