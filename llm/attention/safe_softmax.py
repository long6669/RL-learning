import torch

A = torch.tensor([1., 2., 3., 4., 5.])

def online_softmax(x):
    m = torch.tensor(-1000.0)
    d = 0
    n = len(x)
    a = torch.zeros(n)

    for i in range(n):
        m_pre = m
        m = torch.max(m, x[i])
        d = d * (m_pre-m).exp() + (x[i]-m).exp()

    for i in range(n):
        a[i] = (x[i] - m).exp() / d
    
    return a

print(torch.softmax(A, dim=-1))
print(online_softmax(A))
