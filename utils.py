import os, torch
import torch.distributed as dist
import numpy as np
import random
import datasets
from transformers import TrainingArguments

def SetSeed(key):
    random.seed(key)
    np.random.seed(key)
    torch.manual_seed(key)
    torch.cuda.manual_seed(key)
    torch.cuda.manual_seed_all(key)

class LogLoss:
    def __init__(self, print_steps, local_rank, world_size):
        super().__init__()
        self.log_steps = 0
        self.log_token_loss = 0
        self.log_gate_loss = 0
        self.print_steps = print_steps
        self.local_rank = local_rank
        self.world_size = world_size

    def update(self, token_loss, gate_loss):
        self.log_steps += 1
        self.log_token_loss += token_loss.item()
        self.log_gate_loss += gate_loss.item()
        if self.log_steps % self.print_steps == 0:
            print_token_loss = torch.tensor(self.log_token_loss / self.print_steps).cuda()
            print_gate_loss = torch.tensor(self.log_gate_loss / self.print_steps).cuda()
            if self.world_size > 1:
                dist.reduce(print_token_loss, 0)
                dist.reduce(print_gate_loss, 0)
            if self.local_rank == 0:
                print({'token_loss':print_token_loss.item() / self.world_size, 'gate_loss':print_gate_loss.item() / self.world_size})
            self.log_token_loss = 0
            self.log_gate_loss = 0

def one_hot(indices: torch.Tensor, num_classes: int, unsqueeze_indices=True) -> torch.Tensor:
    if unsqueeze_indices: 
        indices = indices.unsqueeze(-1)
    ret = torch.zeros(indices.shape[:-1] + (num_classes,), device=indices.device, dtype=torch.bool)
    ret.scatter_(-1, indices, 1)
    return ret

def GetDataLoader(data_dir):
    train_dataset = datasets.load_dataset('json', data_files=data_dir, split="train", cache_dir=cache_dir)
    return train_dataset

def GetTrainArgs(args):
    return TrainingArguments(
        bf16=True,
        do_train=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps, 
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        group_by_length=False,
        disable_tqdm=True,
        full_determinism=False,
        log_on_each_node=False,
        ddp_find_unused_parameters=False,
    )

class LlamaMLP(torch.nn.Module):
    def __init__(self, hidden_size, intermediate_size, act_fn):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = act_fn

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

