from dataclasses import dataclass
from simple_parsing.helpers import Serializable
import json
from pathlib import Path



@dataclass
class TrainingArgs(Serializable):
    batch_size: int
    max_seq_len: int
    vocab_source: str 
    vocab_size: int
    gradient_accumulation_steps: int 
    learning_rate: float 
    max_iters: int 
    weight_decay: float 
    beta1: float  
    beta2: float 
    grad_clip: float 
    lr_decay_iter: int 
    min_lr: int
    decay_lr: bool  
    warmup_iters: int 
    device: str
    dtype: str
    compile: bool
    output_dir: str 
    eval_interval: int 
    log_interval: int
    eval_iters: int
    eval_only: bool
    always_save_checkpoint:bool 


@dataclass
class MistralTrainArgs(TrainingArgs):
    pass


MODEL_TYPE_TO_TRAING_ARGS = {
    "generic": TrainingArgs,
    "mistral": MistralTrainArgs,
}


def dispatch_training_args_class(model_type: str):
    return MODEL_TYPE_TO_TRAING_ARGS[model_type]


class TrainingConfig:
    def __init__(self, args: TrainingArgs):
        self.args = args
        for attr, value in vars(args).items():
            setattr(self, attr, value)

    @staticmethod
    def from_json(config_file_path: Path, key: str = "training"):
        with open(config_file_path, "r") as f:
            json_config = json.load(f)
            model_type = json_config["model"]["type"]
            training_args_class = dispatch_training_args_class(model_type)
            training_args = training_args_class.from_dict(json_config["training"])
        return TrainingConfig(training_args)


