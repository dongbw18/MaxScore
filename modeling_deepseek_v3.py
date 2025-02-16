import torch
from transformers import LlamaForCausalLM
from utils import LlamaMLP
from utils import one_hot

class AuxLossFreeTopKGate(torch.nn.Module):
    def __init__(self, hidden_size, n_experts, topk, n_group):
        super().__init__()
        self.n_experts = n_experts
        self.topk = topk
        self.n_group = n_group
        self.wg = torch.nn.Linear(hidden_size, n_experts, bias=False)
        self.score_bias = torch.zeros(n_experts)
        self.u = 1e-3

    def forward(self, x):
        logits = self.wg(x).sigmoid()

        self.score_bias = self.score_bias.to(logits.device)
        gates = logits + self.score_bias * self.u # se

        group_gates = gates.view(gates.shape[0], self.n_group, -1).topk(k=2, dim=2).values.sum(dim=2) # sg
        group_idx = torch.topk(group_gates, k=self.topk // 2, dim=1, sorted=False).indices # s,k/2
        group_mask = torch.zeros_like(group_gates) # sg
        group_mask.scatter_(1, group_idx, 1) # sg
        score_mask = group_mask.unsqueeze(2).expand(gates.shape[0], self.n_group, self.n_experts // self.n_group).reshape(gates.shape[0], self.n_experts) # se

        masked_gates = gates.masked_fill(~score_mask.bool(), 0)
        topk_idx = torch.topk(masked_gates, k=self.topk, dim=-1, sorted=False).indices # sk
        topk_weight = logits.gather(1, topk_idx) # sk
        denom = torch.clamp(topk_weight.sum(dim=1, keepdim=True), min=torch.finfo(gates.dtype).eps) # sk
        topk_weight = topk_weight / denom * 2.5 # sk

        with torch.no_grad():
            assigned_tokens = one_hot(topk_idx.view(-1), self.n_experts).to(logits.dtype).mean(0) # e
            self.score_bias = self.score_bias + torch.sign(assigned_tokens - assigned_tokens.mean())

        return topk_weight.to(logits.dtype), topk_idx

class DeepseekV3MoE(torch.nn.Module):
    def __init__(self, mlp, n_experts=64, topk=6, n_shared=2):
        super().__init__()
        assert n_experts == 64
        assert topk == 6
        assert n_shared == 2
        self.n_experts = n_experts
        self.topk = topk
        self.n_shared = n_shared
        self.hidden_size = mlp.config.hidden_size
        self.intermediate_size = mlp.config.intermediate_size
        self.gate = AuxLossFreeTopKGate(self.hidden_size, n_experts, topk, topk + n_shared)
        self.experts = torch.nn.ModuleList([LlamaMLP(self.hidden_size, self.intermediate_size, mlp.act_fn) for _ in range(n_experts)])
        if n_shared > 0:
            self.shared_expert = LlamaMLP(self.hidden_size, self.intermediate_size * n_shared, mlp.act_fn)

    def forward(self, x):
        reshaped_x = x.reshape(-1, x.shape[-1]) # sm
        topk_weight, topk_idx = self.gate(reshaped_x) # 1, (s, topk), (s, topk)
        reshaped_x = reshaped_x.repeat_interleave(self.topk, dim=0) # (s * topk, m)
        y = torch.empty_like(reshaped_x).to(topk_weight.dtype) # (s * topk, m)
        topk_idx = topk_idx.view(-1) # s * topk
        for i, expert in enumerate(self.experts):
            y[topk_idx == i] = expert(reshaped_x[topk_idx == i])
        y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(1) # (s, topk, m) * (s, topk, 1) -> (s, m)
        y = y.reshape(x.shape)
        if self.n_shared > 0: 
        	y = y + self.shared_expert(x)
        return y # [batch_size, seq_length, hidden_size]

class DeepSeekV3Model(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        for i in range(len(self.model.layers)):
            self.model.layers[i].mlp = DeepseekV3MoE(self.model.layers[i].mlp)
                  