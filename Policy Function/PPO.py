import dis
from operator import is_
import gymnasium as gym 
from mpmath.math2 import EPS
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# ==========================================
# 1. 超参数
# ==========================================
# PPO 特有的参数
K_EPOCHS = 4
EPS_CLIP = 0.2
GAMMA = 0.99
LR_ACTOR = 0.0003
LR_CRITIC = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 记忆库 (Memory)
# ==========================================
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        self.actions.clear()
        self.states.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()

# ==========================================
# 3. Actor-Critic 网络
# ==========================================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    
    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        """选择动作 (用于收集数据)"""
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.item(), action_logprob

    def evaluate(self, state, action):
        """评估动作 (用于更新网络)"""
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy

# ==========================================
# 4. PPO 算法核心
# ========================================== 
class PPO:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': LR_ACTOR},
            {'params': self.policy.critic.parameters(), 'lr': LR_CRITIC}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
    
    def update(self, memory):
        # 1. 计算蒙特卡洛回报 (Returns)
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + GAMMA * discounted_reward
            rewards.insert(0, discounted_reward)
        
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.squeeze(torch.stack(memory.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(memory.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach().to(device)
        
        for _ in range(K_EPOCHS):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            state_values = torch.squeeze(state_values)

            ratio = torch.exp(logprobs - old_logprobs)

            advantages = rewards - state_values.detach()

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-EPS_CLIP, 1+EPS_CLIP) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

        memory.clear_memory()

# ==========================================
# 5. 主循环
# ==========================================    
def train():
    env = gym.make('CartPole-v1', render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim)
    
    print("🚀 开始 PPO 训练...")
    
    running_reward = 0
    time_step = 0
    
    # PPO 是 On-Policy，需要每隔一定步数更新一次
    UPDATE_TIMESTEP = 2000 
    
    for i_episode in range(1, 10000):
        state, _ = env.reset()
        ep_reward = 0
        
        for t in range(1000):
            time_step += 1
            
            # 使用旧策略收集数据
            state_tensor = torch.FloatTensor(state).to(device)
            action, logprob = ppo.policy_old.act(state_tensor)
            
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 存入记忆
            memory.states.append(state_tensor)
            memory.actions.append(torch.tensor(action))
            memory.logprobs.append(logprob)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            ep_reward += reward
            
            # 如果收集的数据够多了，就开始学习 (Update)
            if time_step % UPDATE_TIMESTEP == 0:
                ppo.update(memory)
                time_step = 0
            
            if done:
                break
                
        running_reward += ep_reward
        
        # 打印日志
        if i_episode % 100 == 0:
            avg_reward = running_reward / 20
            print(f'Episode {i_episode} \t Avg Reward: {avg_reward:.2f}')
            running_reward = 0
            
            if avg_reward > 495:
                print("✅ PPO 训练完成！策略已收敛。")
                break
    
    env.close()
    return ppo

if __name__ == '__main__':
    trained_ppo = train()
    
    # 展示效果
    print("\n🎥 展示 PPO 效果...")
    env = gym.make('CartPole-v1', render_mode='human')
    state, _ = env.reset()
    for _ in range(1000):
        state_tensor = torch.FloatTensor(state).to(device)
        action, _ = trained_ppo.policy_old.act(state_tensor)
        state, reward, done, truncated, _ = env.step(action)
        if done or truncated:
            break
    env.close()

