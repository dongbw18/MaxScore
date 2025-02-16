import os
import torch
from transformers import LlamaForCausalLM
import torch.nn.functional as F
from typing import Tuple
from torch import Tensor
from utils import one_hot
from tutel import moe as tutel_moe
from utils import LlamaMLP
import numpy as np
from ctypes import cdll, c_void_p

class OursTop2Gate(torch.nn.Module):
    def __init__(self, model_dim: int, num_experts: int) -> None:
        super().__init__()
        self.Wg = torch.nn.Linear(model_dim, num_experts, bias=False)
        self.lib = cdll.LoadLibrary('./KM.so')
        self.lib.KM.argtypes = [c_void_p, c_void_p, c_void_p]
        self.seq_len = 512
        self.steps = 0

    def calc(self, scores_gpu):
        scores_cpu = scores_gpu.view(-1).to(torch.float32).cpu().detach().numpy()
        indices1_s_cpu = torch.zeros(scores_gpu.shape[0]).to(torch.int32).numpy()
        indices2_s_cpu = torch.zeros(scores_gpu.shape[0]).to(torch.int32).numpy()
        self.lib.KM(scores_cpu.ctypes.data_as(c_void_p), indices1_s_cpu.ctypes.data_as(c_void_p), indices2_s_cpu.ctypes.data_as(c_void_p))
        return torch.from_numpy(indices1_s_cpu).to(scores_gpu.device).to(torch.int64), torch.from_numpy(indices2_s_cpu).to(scores_gpu.device).to(torch.int64)

    def forward(self, x: torch.Tensor) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore
        logits = self.Wg(x) # se

        num_tokens, num_experts = logits.shape[0], logits.shape[1]
        capacity = 2 * num_tokens // num_experts
        assert (2 * num_tokens) % num_experts == 0

        gates = F.softmax(logits, dim=1, dtype=torch.float) # se
        
        indices1_list, indices2_list = [], []
        for b in range(gates.shape[0] // self.seq_len):
            i1, i2 = self.calc(gates[b * self.seq_len: (b + 1) * self.seq_len])
            indices1_list.append(i1)
            indices2_list.append(i2)
        indices1_s, indices2_s = torch.cat(indices1_list, dim=0), torch.cat(indices2_list, dim=0) # s, s

        mask1 = one_hot(indices1_s, num_experts) # se
        locations1 = torch.cumsum(mask1, dim=0) - 1 # se
        mask2 = one_hot(indices2_s, num_experts) # se
        locations2 = torch.cumsum(mask2, dim=0) - 1 + torch.sum(mask1, dim=0, keepdim=True) # se

        locations1_s = torch.sum(locations1 * mask1, dim=1) # s
        locations2_s = torch.sum(locations2 * mask2, dim=1) # s

        l_aux = torch.mean(gates.mean(0) * mask1.float().mean(0)) # 1

        self.steps += 1
        t = max(4 - self.steps / 3e4 * 3, 1)
        gates1_s = (gates * mask1).sum(dim=1) # s
        gates2_s = (gates * mask2).sum(dim=1) * t # s

        return l_aux.to(logits.dtype), capacity, num_experts, [indices1_s, indices2_s], [locations1_s, locations2_s], [gates1_s, gates2_s]

class OursTutelMoE(torch.nn.Module):
    def __init__(self, mlp, n_experts=16, topk=2):
        super().__init__()
        assert topk == 2
        assert n_experts == 16
        self.n_experts = n_experts
        self.hidden_size = mlp.config.hidden_size
        self.intermediate_size = mlp.config.intermediate_size
        self.gate = OursTop2Gate(self.hidden_size, self.n_experts)
        self.experts = torch.nn.ModuleList([LlamaMLP(self.hidden_size, self.intermediate_size, mlp.act_fn) for _ in range(n_experts)])
        self.l_aux = 0
        self.loss_coef = 1e-2 * n_experts * n_experts

    def forward(self, x):
        hidden_size = x.shape[-1]
        reshaped_x = x.reshape(-1, hidden_size) # sm
        l_aux, capacity, num_experts, indices_list, location_list, weight_list = self.gate(reshaped_x) # 1, 1, 1, ks, ks, ks
        self.l_aux += l_aux * self.loss_coef
        if not hasattr(self, "_tutel_dispatcher"):
            self._tutel_dispatcher = tutel_moe.fast_dispatcher(num_experts, capacity, hidden_size, dispatch_dtype=reshaped_x.dtype)
        self._tutel_dispatcher.update(indices_list, location_list, weight_list, capacity=capacity)
        dispatched_input = self._tutel_dispatcher.encode(reshaped_x).reshape(num_experts, capacity, hidden_size) # ecm
        expert_outputs = torch.cat([expert(chunk).unsqueeze(0) for chunk, expert in zip(dispatched_input, self.experts)], dim=0) # ecm
        combined_outputs = self._tutel_dispatcher.decode(expert_outputs.view(-1, hidden_size)) # sm
        return combined_outputs.reshape(x.shape) # [batch_size, seq_length, hidden_size]

class OurModel(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        for i in range(len(self.model.layers)):
            self.model.layers[i].mlp = OursTutelMoE(self.model.layers[i].mlp)
