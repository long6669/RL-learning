import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# ==========================================
# 1. 定义 Actor-Critic 网络
# ==========================================
class ActorCritic(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.affine = nn.Linear(n_observations, 128)

        self.action = nn.Linear(128, n_actions)

        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.affine(x))

        action_prob = F.softmax(self.action(x), dim=-1)

        state_values = self.value(x)

        return action_prob, state_values

# ==========================================
# 2. 环境与超参数
# ==========================================
env = gym.make("CartPole-v1", render_mode=None)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ActorCritic(env.observation_space.shape[0], env.action_space.n).to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = torch.finfo(torch.float32).eps

def select_action(state):
    state = torch.from_numpy(state).float().to(device)
    probs, state_value = model(state)

    m = Categorical(probs)
    action = m.sample()

    return action.item(), m.log_prob(action), state_value

def finish_episode(saved_log_probs, saved_values, rewards):
    R = 0
    policy_losses = []
    value_losses = []
    returns = []

    for r in rewards[::-1]:
        R = r + 0.99 * R
        returns.insert(0, R)
    
    returns = torch.tensor(returns).to(device)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for log_prob, value, R in zip(saved_log_probs, saved_values, returns):
        advantage = R - value.item()
        policy_losses.append(-log_prob * advantage)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]).to(device)))

    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    optimizer.step()

# ==========================================
# 3. 主循环
# ==========================================
print("🚀 开始 Actor-Critic 训练...")

for i_episode in range(1000):
    state, _ = env.reset()
    ep_reward = 0

    saved_log_probs = []
    saved_values = []
    rewards = []

    for t in range(10000):
        action, log_prob, value = select_action(state)

        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        saved_log_probs.append(log_prob)
        saved_values.append(value)
        rewards.append(reward)

        ep_reward += reward

        if done:
            break

    finish_episode(saved_log_probs, saved_values, rewards)

    if i_episode % 100 == 0:
        print(f'Episode {i_episode}\tReward: {ep_reward}')
    
    if ep_reward > 495:
        print(f"✅ 解决了！Last Reward: {ep_reward}")
        break

env.close()

# ==========================================
# 4. 效果展示
# ==========================================
print("\n🎥 展示 Actor-Critic 效果...")
env_test = gym.make("CartPole-v1", render_mode="human")
state, _ = env_test.reset()
done = False
while not done:
    state = torch.from_numpy(state).float().to(device)
    # 只要 probabilities，不需要 value
    probs, _ = model(state)
    action = probs.argmax().item()
    state, _, terminated, truncated, _ = env_test.step(action)
    done = terminated or truncated
env_test.close()
