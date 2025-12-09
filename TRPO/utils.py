from mpmath import residual
import torch
import torch.nn.functional as F
import torch.autograd as autograd

def compute_GAE(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    """
    计算 GAE (广义优势估计)
    
    参数:
        next_value: 轨迹最后一步状态的价值 V(s_last)
        rewards:    每一步的奖励列表
        masks:      每一步的掩码 (1代表未结束，0代表结束)
        values:     每一步的状态价值列表 V(s)
        gamma:      折扣因子 (通常 0.99)
        tau:        GAE 参数 (通常 0.95，用于平滑)
        
    返回:
        returns:    真实的折扣回报 (用于训练 ValueNet)
        advantages: 动作优势 (用于训练 PolicyNet)
    """
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step+1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def update_value_net(value_net, value_optimizer, states, targets):
    """
    训练价值网络 (Critic)
    简单的回归任务：让 V(s) 尽可能接近计算出的 returns
    """
    values_pred = value_net(states)
    value_loss = F.mse_loss(values_pred, targets.view(-1, 1))
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    return value_loss.item()

def compute_grad(policy_net, states, actions, advantages, old_log_probs):
    """
    计算目标函数 (Surrogate Objective) 的梯度 g
    """
    log_probs = policy_net.get_log_prob(states, actions)
    ratio = torch.exp(log_probs - old_log_probs)

    objective = (ratio * advantages).mean()
    grads = autograd.grad(objective, policy_net.parameters())
    flag_grad = torch.cat([grad.view(-1) for grad in grads]).detach()

    return flag_grad

def Hession_Vector_Product(policy_net, states, v, damping=0.1):
    """
    【核心】计算 Hessian 矩阵与向量 v 的乘积: H * v
    利用 Autograd 技巧 (Pearlmutter's trick)
    """
    kl = policy_net.get_kl(states)
    kl = kl.mean()

    grads = autograd.grad(kl, policy_net.parameters(), create_graph=True)
    flag_grad_kl = torch.cat([grad.view(-1) for grad in grads])

    kl_v = (flag_grad_kl * v).sum()

    grads_hvp = autograd.grad(kl_v, policy_net.parameters())
    flat_grad_hvp = torch.cat([grad.contiguous().view(-1) for grad in grads_hvp])

    return flat_grad_hvp + v * damping

def Conjugate_Gradient(policy_net, states, b, n_steps=10, residual_tol=1e-10):
    """
    共轭梯度法 (CG) 求解线性方程: Hx = b
    这里 b 就是梯度 g，我们要解出 x (搜索方向)
    """
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rdotr = torch.dot(r, r)
    for _ in range(n_steps):
        hvp = Hession_Vector_Product(policy_net, states, p)

        alpha = rdotr / torch.dot(p, hvp)

        x += alpha * p

        r -= alpha * hvp

        new_rdotr = torch.dot(r, r)
        if new_rdotr < residual_tol:
            break

        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr

    return x
    

