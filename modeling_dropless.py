import torch
from transformers import LlamaForCausalLM
from utils import LlamaMLP
from utils import one_hot

class Top2Gate(torch.nn.Module):
    def __init__(self, hidden_size, n_experts):
        super().__init__()
        self.n_experts = n_experts
        self.wg = torch.nn.Linear(hidden_size, n_experts, bias=False)

    def forward(self, x):
        logits = self.wg(x)

        gates = torch.nn.functional.softmax(logits, dim=1, dtype=torch.float) # se
        topk_weight, topk_idx = torch.topk(gates, k=2, dim=-1, sorted=False) # sk

        l_aux = torch.mean(gates.mean(0) * one_hot(torch.argmax(gates, dim=1), self.n_experts).float().mean(0)) # 1

        return l_aux.to(x.dtype), topk_weight.to(logits.dtype), topk_idx

class DropLessMoE(torch.nn.Module):
    def __init__(self, mlp, n_experts=16, topk=2):
        super().__init__()
        assert topk == 2
        assert n_experts == 16
        self.n_experts = n_experts
        self.hidden_size = mlp.config.hidden_size
        self.intermediate_size = mlp.config.intermediate_size
        self.gate = Top2Gate(self.hidden_size, n_experts)
        self.experts = torch.nn.ModuleList([LlamaMLP(self.hidden_size, self.intermediate_size, mlp.act_fn) for _ in range(n_experts)])
        self.l_aux = 0
        self.loss_coef = 1e-3

    def forward(self, x):
        reshaped_x = x.reshape(-1, x.shape[-1]) # sm
        l_aux, topk_weight, topk_idx = self.gate(reshaped_x) # 1, (s, topk), (s, topk)
        self.l_aux += l_aux * self.loss_coef
        reshaped_x = reshaped_x.repeat_interleave(2, dim=0) # (s * topk, m)
        y = torch.empty_like(reshaped_x).to(topk_weight.dtype) # (s * topk, m)
        topk_idx = topk_idx.view(-1) # s * topk
        for i, expert in enumerate(self.experts):
            y[topk_idx == i] = expert(reshaped_x[topk_idx == i])
        y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(1) # (s, topk, m) * (s, topk, 1) -> (s, m)
        y = y.reshape(x.shape)
        return y # [batch_size, seq_length, hidden_size]

class DropLessModel(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        for i in range(len(self.model.layers)):
            self.model.layers[i].mlp = DropLessMoE(self.model.layers[i].mlp)
                  