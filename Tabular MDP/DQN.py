import gymnasium as gym 
import math
import random
import matplotlib.pyplot as plt
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.quantization import no_observer_set

# ==========================================
# 1. 定义神经网络 (Q-Network)
# ==========================================
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

# ==========================================
# 2. 超参数设置
# ==========================================
BATCH_SIZE = 128        # 每次从回放池取多少条数据来训练
GAMMA = 0.99            # 折扣因子
EPS_START = 0.9         # 初始探索率
EPS_END = 0.05          # 最终探索率
EPS_DECAY = 1000        # 衰减速度
TAU = 0.005             # 目标网络软更新系数 (每次只学一点点)
LR = 1e-4               # 学习率

# 3.创建环境
env = gym.make("CartPole-v1", render_mode=None)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_actions = env.action_space.n
state, _ = env.reset()
n_observations = len(state)

# 4.初始化两个网络
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = deque(maxlen=10000)

steps_done = 0

# ==========================================
# 5. 辅助函数
# ==========================================
def select_action(state):
    """Epsilon-Greedy 策略 (带神经网络版)"""
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device = device, dtype=torch.long)

def optimize_model():
    """DQN 的训练步骤"""
    if len(memory) < BATCH_SIZE:
        return 
    transitions = random.sample(memory, BATCH_SIZE)
    batch_state, batch_action, batch_next_state, batch_reward, batch_done = zip(*transitions)

    state_batch = torch.cat(batch_state)
    action_batch = torch.cat(batch_action)
    next_state_batch = torch.cat(batch_next_state)
    reward_batch = torch.cat(batch_reward)
    done_batch = torch.cat(batch_done)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    with torch.no_grad():
        next_state_values = target_net(next_state_batch).max(1)[0]
        next_state_values = next_state_values * (1 - done_batch)
    
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# ==========================================
# 6. 训练主循环
# ==========================================
num_episodes = 300
print(f"🚀 开始 DQN 训练 CartPole，共 {num_episodes} 局...")

episode_durations = []
for i_episode in range(num_episodes):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    for t in range(1000):
        action = select_action(state)
        observation, reward, terminated, truncted, _ = env.step(action.item())
        done = terminated or truncted

        reward = torch.tensor([reward], device=device)
        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        done_flag = torch.tensor([1.0 if done else 0.0], device=device)

        memory.append((state, action, next_state, reward, done_flag))

        state = next_state

        optimize_model()

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t+1)
            break
    if (i_episode + 1) % 50 == 0:
        print(f"Episode {i_episode+1}, 坚持时长: {t+1} 步")

print("✅ 训练完成！")
env.close()

# ==========================================
# 7. 成果展示 (看画面)
# ==========================================
print("\n🎥 展示模型效果...")
env_test = gym.make("CartPole-v1", render_mode="human")
state, _ = env_test.reset()
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

done = False
while not done:
    with torch.no_grad():
        # 完全贪婪
        action = policy_net(state).max(1)[1].view(1, 1)
    
    observation, _, terminated, truncated, _ = env_test.step(action.item())
    done = terminated or truncated
    state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

env_test.close()