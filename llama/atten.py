import torch

def repeat_kv(x: torch.Tensor, n_rep:int) -> torch.Tensor:

    batch_size, seq_len, n_kv_heads, head_dim = x.shape

    if n_rep == 1:
        return x
    
    return (x[:, :, :, None, :]
    .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
    .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    