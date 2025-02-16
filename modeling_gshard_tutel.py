import copy
import torch
from transformers import LlamaForCausalLM
import torch.nn.functional as F
from typing import Tuple
from torch import Tensor
from utils import one_hot
from tutel import moe as tutel_moe
from utils import LlamaMLP

class TutelTop2Gate(torch.nn.Module):
    def __init__(self, model_dim: int, num_experts: int) -> None:
        super().__init__()
        self.Wg = torch.nn.Linear(model_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore
        logits = self.Wg(x) # se

        num_tokens, num_experts = logits.shape[0], logits.shape[1]
        capacity = 2 * num_tokens // num_experts
        assert (2 * num_tokens) % num_experts == 0

        gates = F.softmax(logits, dim=1, dtype=torch.float) # se

        indices1_s = torch.argmax(gates, dim=1) # s
        mask1 = one_hot(indices1_s, num_experts) # se
        logits_except1 = logits.masked_fill(mask1.bool(), float("-inf")) # se

        l_aux = torch.mean(gates.mean(0) * mask1.float().mean(0)) # 1

        locations1 = torch.cumsum(mask1, dim=0) - 1 # se
        mask1 *= torch.lt(locations1, capacity) # se
        indices1_s = indices1_s.masked_fill(~mask1.sum(1).bool(), -1) # se

        indices2_s = torch.argmax(logits_except1, dim=1) # s
        mask2 = one_hot(indices2_s, num_experts) # se
        locations2 = torch.cumsum(mask2, dim=0) - 1 + torch.sum(mask1, dim=0, keepdim=True) # se
        mask2 *= torch.lt(locations2, capacity) # se
        indices2_s = indices2_s.masked_fill(~mask2.sum(1).bool(), -1) # s

        locations1_s = torch.sum(locations1 * mask1, dim=1) # s
        locations2_s = torch.sum(locations2 * mask2, dim=1) # s

        gates1_s = (gates * mask1).sum(dim=1) # s
        gates2_s = (gates * mask2).sum(dim=1) # s

        return l_aux.to(logits.dtype), capacity, num_experts, [indices1_s, indices2_s], [locations1_s, locations2_s], [gates1_s, gates2_s]

class GshardTutelMoE(torch.nn.Module):
    def __init__(self, mlp, n_experts=16, topk=2):
        super().__init__()
        assert topk == 2
        assert n_experts == 16
        self.n_experts = n_experts
        self.hidden_size = mlp.config.hidden_size
        self.intermediate_size = mlp.config.intermediate_size
        self.gate = TutelTop2Gate(self.hidden_size, self.n_experts)
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

class GShardTutelModel(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        for i in range(len(self.model.layers)):
            self.model.layers[i].mlp = GshardTutelMoE(self.model.layers[i].mlp)
