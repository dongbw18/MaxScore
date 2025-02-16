import torch
from transformers import LlamaForCausalLM
from utils import LlamaMLP

class TopKGateGroup(torch.nn.Module):
    def __init__(self, hidden_size, n_experts, topk, n_group):
        super().__init__()
        self.n_experts = n_experts
        self.topk = topk
        self.n_group = n_group
        self.wg = torch.nn.Linear(hidden_size, n_experts, bias=False)

    def forward(self, x):
        logits = self.wg(x)

        gates = torch.nn.functional.softmax(logits, dim=1, dtype=torch.float) # se
        group_gates = gates.view(gates.shape[0], self.n_group, -1).max(dim=2).values # sg
        group_idx = torch.topk(group_gates, k=self.topk // 2, dim=1, sorted=False).indices # s,k/2
        group_mask = torch.zeros_like(group_gates) # sg
        group_mask.scatter_(1, group_idx, 1) # sg
        score_mask = group_mask.unsqueeze(2).expand(gates.shape[0], self.n_group, self.n_experts // self.n_group).reshape(gates.shape[0], self.n_experts) # se

        masked_gates = gates.masked_fill(~score_mask.bool(), 0)
        topk_weight, topk_idx = torch.topk(masked_gates, k=self.topk, dim=-1, sorted=False) # sk
        topk_weight *= 16

        bsz, seq_len = x.shape[0] // 512, 512
        me = gates.view(bsz, seq_len, -1).mean(dim=1) # bne->be
        ce = torch.zeros(bsz, self.n_experts).to(gates.device) # be
        ce.scatter_add_(1, topk_idx.reshape(bsz, -1), torch.ones(bsz, seq_len * self.topk).to(gates.device)).div_(seq_len * self.topk / self.n_experts)
        l_aux = torch.mean((me * ce).sum(dim=1))

        return l_aux.to(x.dtype), topk_weight.to(logits.dtype), topk_idx

class DeepseekV2MoE(torch.nn.Module):
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
        self.gate = TopKGateGroup(self.hidden_size, n_experts, topk, topk + n_shared)
        self.experts = torch.nn.ModuleList([LlamaMLP(self.hidden_size, self.intermediate_size, mlp.act_fn) for _ in range(n_experts)])
        if n_shared > 0:
            self.shared_expert = LlamaMLP(self.hidden_size, self.intermediate_size * n_shared, mlp.act_fn)
        self.l_aux = 0
        self.loss_coef = 1e-3

    def forward(self, x):
        reshaped_x = x.reshape(-1, x.shape[-1]) # sm
        l_aux, topk_weight, topk_idx = self.gate(reshaped_x) # 1, (s, topk), (s, topk)
        self.l_aux += l_aux * self.loss_coef
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

class DeepSeekModel(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        for i in range(len(self.model.layers)):
            self.model.layers[i].mlp = DeepseekV2MoE(self.model.layers[i].mlp)
                  