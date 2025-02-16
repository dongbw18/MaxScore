import torch
from transformers import LlamaForCausalLM
import torch.nn.functional as F
from typing import Tuple
from torch import Tensor
from utils import LlamaMLP

class ExpertChoiceTop2Gate(torch.nn.Module):
    def __init__(self, model_dim: int, num_experts: int) -> None:
        super().__init__()
        self.Wg = torch.nn.Linear(model_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore
        logits = self.Wg(x) # se

        num_tokens, num_experts = logits.shape[0], logits.shape[1]
        capacity = 2 * num_tokens // num_experts
        assert (2 * num_tokens) % num_experts == 0

        gates = F.softmax(logits, dim=1, dtype=torch.float) # se
        topc_weights, topc_indices = torch.topk(gates.T, k=capacity, dim=1) # ec, ec

        mask = torch.zeros(logits.shape[0]).to(torch.int32).to(topc_indices.device)
        for indice in topc_indices: mask[indice] += 1
        print([(mask == i).sum().cpu().item() for i in range(17)])

        l_aux = torch.mean(gates.mean(0) ** 2)

        return l_aux.to(logits.dtype), topc_weights.to(logits.dtype), topc_indices

class ExpertChoiceMoE(torch.nn.Module):
    def __init__(self, mlp, n_experts=16, topk=2):
        super().__init__()
        assert topk == 2
        assert n_experts == 16
        self.n_experts = n_experts
        self.hidden_size = mlp.config.hidden_size
        self.intermediate_size = mlp.config.intermediate_size
        self.gate = ExpertChoiceTop2Gate(self.hidden_size, self.n_experts)
        self.experts = torch.nn.ModuleList([LlamaMLP(self.hidden_size, self.intermediate_size, mlp.act_fn) for _ in range(n_experts)])
        self.l_aux = 0
        self.loss_coef = 1e-2 * n_experts * n_experts

    def forward(self, x):
        reshaped_x = x.reshape(-1, x.shape[-1]) # sm
        l_aux, topc_weights, topc_indices = self.gate(reshaped_x) # 1, ec, ec
        self.l_aux += l_aux * self.loss_coef
        y = torch.zeros_like(reshaped_x) # sm
        for e, expert in enumerate(self.experts):
            y[topc_indices[e]] += topc_weights[e].unsqueeze(1) * expert(reshaped_x[topc_indices[e]])
        return y.reshape(x.shape) # [batch_size, seq_length, hidden_size]

class ExpertChoiceModel(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        for i in range(len(self.model.layers)):
            self.model.layers[i].mlp = ExpertChoiceMoE(self.model.layers[i].mlp)
    