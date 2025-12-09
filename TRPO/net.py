import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.mu_head = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))

        mu = self.mu_head(x)
        std = torch.exp(self.log_std.expand_as(mu))

        return mu, std
    
    def get_distribution(self, x):
        """辅助函数：返回正态分布对象"""
        mu, std = self.forward(x)
        return Normal(mu, std)

    def get_log_prob(self, x, actions):
        """计算动作的对数概率 (用于计算 Loss)"""
        dist = self.get_distribution(x)
        return dist.log_prob(actions).sum(dim=-1)

    def get_kl(self, x):
        """
        【TRPO 核心函数】计算 KL(Old || New)
        用于 Fisher 向量乘积 (HVP) 的计算
        """
        mu, std = self.forward(x)
        new_dist = Normal(mu, std)

        old_mu = mu.detach()
        old_std = std.detach()
        old_dist = Normal(old_mu, old_std)

        kl = torch.distributions.kl.kl_divergence(old_dist, new_dist)
        return kl.sum(dim=-1)
    
class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.head(x)