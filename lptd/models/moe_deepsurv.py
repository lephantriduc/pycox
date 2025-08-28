import torch
import torchtuples as tt
import torch.nn as nn
import torch.nn.functional as F

# Implementation of Mixture of Experts on DeepSurv
class MoEDeepSurv(nn.Module):
    def __init__(self,
                 input_dim,
                 n_experts=3,
                 expert_hidden=[64, 32],
                 gate_hidden=[32],
                 expert_dropout=0.3,
                 gate_dropout=0.1):
        super().__init__()
        self.n_experts = n_experts

        # Experts
        self.experts = nn.ModuleList([
            self._make_expert(input_dim, expert_hidden, expert_dropout)
            for _ in range(n_experts)
        ])

        # Gate
        self.gate = self._make_gate(input_dim, gate_hidden, gate_dropout, n_experts)

    def _make_expert(self, input_dim, hidden_layers, dropout):
        layers = []
        dims = [input_dim] + hidden_layers
        for i in range(len(hidden_layers)):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], 1))  # log-hazard
        return nn.Sequential(*layers)

    def _make_gate(self, input_dim, hidden_layers, dropout, n_experts):
        layers = []
        dims = [input_dim] + hidden_layers
        for i in range(len(hidden_layers)):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], n_experts))  # logits for experts
        return nn.Sequential(*layers)

    def forward(self, x):
        gate_logits = self.gate(x)                # (batch, n_experts)
        gate_w = F.softmax(gate_logits, dim=1)    # mixture weights

        expert_outs = torch.cat([expert(x) for expert in self.experts], dim=1)  # (batch, n_experts)
        log_h = (gate_w * expert_outs).sum(dim=1)                         # (batch,)

        return log_h
