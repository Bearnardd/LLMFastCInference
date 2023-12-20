from dataclasses import dataclass, fields, asdict
from simple_parsing.helpers import Serializable
from tf_blocks import MistralBlock, MODEL_TYPE_TO_BLOCK_TYPE
from model_args import ModelArgs, MODEL_TYPE_TO_MODEL_ARGS, BASE_MODEL_ARG_NAMES
# TODO: change that
from modules.common import NORM_STR_TO_NORM_NAME_AND_ARGS, NORM_STR_TO_NORM

from pathlib import Path
import json

import torch.nn as nn
import torch

import inspect


class GenericTransformerModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        # TODO: change that
        tf_block_args_names = self._get_tf_block_args_names(args)
        tf_block_args = {key: value for key, value in asdict(args).items() if key in tf_block_args_names}
        TransformerBlock = MODEL_TYPE_TO_BLOCK_TYPE[args.type]
        self.layers = nn.ModuleList(
            [TransformerBlock(**tf_block_args) for _ in range(args.n_layers)]
        )
        if args.output_layers:
            self.output_layers = self._add_output_layers(args)
        # TODO: needed?
        # for attr, value in vars(args).items():
        #     setattr(self, attr, value)

    @staticmethod
    def _get_tf_block_args_names(args: ModelArgs):
        model_args_class = MODEL_TYPE_TO_MODEL_ARGS[args.type]
        tf_block_arg_names = [field.name for field in fields(model_args_class) if field.name not in BASE_MODEL_ARG_NAMES]
        return tf_block_arg_names

    def _add_output_layers(self, args):
        output_layers = []
        layers_names = args.output_layers.split("->")
        for layer_name in layers_names:
            if NORM_STR_TO_NORM.get(layer_name): 
                norm_class, required_args = NORM_STR_TO_NORM_NAME_AND_ARGS[layer_name]
                for arg in required_args:
                    assert hasattr(args, arg), f"Missing required arg {arg} for {norm_class}"
                norm_args = {arg: getattr(args, arg) for arg in required_args} 
                norm_layer = norm_class(**norm_args)
                output_layers.append(norm_layer)
            if "linear" in layer_name:
                # TODO: make it betterm, it is baaad :)
                if "(" in layer_name or "[" in layer_name:
                    beg_sep, end_sep = ["(", ")"] if "(" in layer_name else ["[", "]"]
                    start = layer_name.find(beg_sep) + 1  # Find the index of the opening parenthesis
                    end = layer_name.find(end_sep)  # Find the index of the closing parenthesis
                    in_shape, out_shape = list(map(int, layer_name[start:end].split(',')))  # Extract the values and convert them to integers
                else:
                    # single output layer
                    in_shape, out_shape = args.dim, args.vocab_size
                output_layers.append(nn.Linear(in_shape, out_shape))
        return nn.ModuleList(output_layers)
                # layers.append()

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        # TODO : add more optimizers
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    @staticmethod
    def from_json(config_file_path: Path):
        with open(config_file_path, "r") as f:
            json_config = json.load(f)
            model_type = json_config["model"]["type"]
            # TODO it should not be here -> training/model should be separated
            device = json_config["training"]["device"]
            model_args_class = MODEL_TYPE_TO_MODEL_ARGS[model_type]
            model_args = model_args_class.from_dict(json_config["model"])
        return GenericTransformerModel(model_args).to(device)