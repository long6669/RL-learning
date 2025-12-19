import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# ==========================================
# 1. 定义策略网络 (Policy Network)
# ==========================================
class policy_net(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(n_observations, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.fc2 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return F.softmax(x, dim=1)

# ==========================================
# 2. 超参数与环境
# ==========================================
env = gym.make("CartPole-v1", render_mode=None)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n

policy = policy_net(n_observations, n_actions).to(device)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
gamma = 0.99

saved_log_probs = []
rewards = []

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    probs = policy(state)

    m = Categorical(probs)
    action = m.sample()

    saved_log_probs.append(m.log_prob(action))

    return action.item()

def finish_episode():
    R = 0
    policy_loss = []
    returns = []

    for r in rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)

    for log_prob, R in zip(saved_log_probs, returns):
        policy_loss.append(-log_prob * R)

    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()

    del saved_log_probs[:]
    del rewards[:]

# ==========================================
# 3. 主循环
# ==========================================
print("🚀 开始 REINFORCE 训练...")
running_reward = 10   

for i_episode in range(1000):
    state, _ = env.reset()
    ep_reward = 0
    
    # 玩一整局 (蒙特卡洛风格)
    for t in range(10000): 
        action = select_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        rewards.append(reward)
        ep_reward += reward
        if done:
            break
            
    # 玩完一局，马上学习
    finish_episode()
    
    # 平滑显示的奖励
    running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
    
    if i_episode % 20 == 0:
        print(f'Episode {i_episode}\tLast Reward: {ep_reward:.2f}\tAverage Reward: {running_reward:.2f}')
        
    if running_reward > env.spec.reward_threshold:
        print(f"✅ 解决了！Running reward is now {running_reward:.2f}")
        break

env.close()

# ==========================================
# 4. 展示效果
# ==========================================
print("\n🎥 展示 REINFORCE 效果...")
env_test = gym.make("CartPole-v1", render_mode="human")
state, _ = env_test.reset()

done = False
while not done:
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    probs = policy(state)
    action = probs.argmax().item() # 展示时直接选概率最大的，不抽样了
    
    state, _, terminated, truncated, _ = env_test.step(action)
    done = terminated or truncated

env_test.close()