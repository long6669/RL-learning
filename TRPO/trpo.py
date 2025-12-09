import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from utils import compute_grad, Conjugate_Gradient, Hession_Vector_Product
import torch.nn.utils as utils

def set_flat_params_to_model(model, flat_params):
    """辅助函数：将扁平化的参数向量赋值给模型"""
    utils.vector_to_parameters(flat_params, model.parameters())
    
def get_flat_params_from_model(model):
    """辅助函数：获取模型当前的参数向量 (扁平化)"""
    return utils.parameters_to_vector(model.parameters())

def trpo_step(policy_net, states, actions, advantages, old_log_probs,
    old_mus, old_stds, damping=0.1, max_kl=0.01) :
    """
    【TRPO 主控函数】执行一次完整的策略更新
    """
    # --- 1. 计算搜索方向 (Conjugate Gradient) ---
    # 计算目标函数梯度 g
    g = compute_grad(policy_net, states, actions, advantages, old_log_probs)

    # 使用 CG 计算 H^-1 * g
    # hession_vector_product 会在内部被调用
    search_dir = Conjugate_Gradient(policy_net, states, g)

    # --- 2. 计算最大步长 (Lagrange Multiplier) ---
    # beta = sqrt( 2 * max_kl / (x^T * H * x) )
    # 这里再次计算一次 HVP: H * step_dir
    H_search_dir = Hession_Vector_Product(policy_net, states, search_dir)

    shs = (search_dir * H_search_dir).sum(dim=0, keepdim=True)

    if shs < 0:
        print("Hessian is not positive definite, skipping update.")
        return
    
    max_step = torch.sqrt(2 * max_kl / shs)
    full_step = max_step * search_dir

    # --- 3. 线性搜索 (Line Search) ---
    
    # 保存当前的旧参数 (Backup)
    old_params = get_flat_params_from_model(policy_net)

    with torch.no_grad():
        current_log_probs = policy_net.get_log_prob(states, actions)
        ratio = torch.exp(current_log_probs - old_log_probs)
        old_loss = (ratio *  advantages).mean()

    expected_improve = (g * full_step).sum()

    # 开始尝试步长: 1.0, 0.5, 0.25 ...
    for i in range(10):
        step_frac = 0.5 ** i
        new_params = old_params + step_frac * full_step
        
        # 将新参数写入网络
        set_flat_params_to_model(policy_net, new_params)
        
        # --- 4. 检查约束 (Constraint Check) ---
        with torch.no_grad():
            # a. 计算新的 Loss
            new_log_probs = policy_net.get_log_prob(states, actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            new_loss = (ratio * advantages).mean()
            
            # b. 计算真实的 KL (Old Data vs New Net)
            # 这里必须用 Part 1 里没有用到的 old_mus/old_stds (采样数据)
            new_mu, new_std = policy_net(states)
            
            dist_old = Normal(old_mus, old_stds) # 采样时的分布
            dist_new = Normal(new_mu, new_std)   # 新参数下的分布
            kl = torch.distributions.kl.kl_divergence(dist_old, dist_new).mean()

        # --- 5. 判定条件 ---
        # 条件1: KL <= delta (加上一点点容错率)
        # 条件2: Loss 必须提升 (new_loss > old_loss)
        if kl <= max_kl * 1.5 and new_loss > old_loss:
            print(f"Update accepted at step {i}! KL: {kl.item():.4f}, Loss Improv: {new_loss - old_loss:.4f}")
            return # 更新成功，直接退出
            
    # 如果循环走完都没成功
    print("Update rejected. Restoring old parameters.")
    set_flat_params_to_model(policy_net, old_params)




    




     




