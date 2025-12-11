import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dims = hidden_size // num_heads

        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)

        self.out = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, hidden_state, attention_mask=None):
        batch_size = hidden_state.size()[0]

        query = self.q(hidden_state)
        key = self.k(hidden_state)
        value = self.v(hidden_state)

        m_query = self.split_head(query)
        m_key = self.split_head(key)
        m_value = self.split_head(value)

        attention_scores = torch.matmul(m_query, m_key.transpose(-1, -2)) / math.sqrt(self.head_dims)

        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask==0, float('-inf'))

        attention_probs = torch.softmax(attention_scores, dim=-1)

        output = torch.matmul(attention_probs, m_value)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dims)
        output = self.out(output)

        return output


    def split_head(self, x):
        batch_size = x.size()[0]
        return x.view(batch_size, -1, self.num_heads, self.head_dims).transpose(1, 2)