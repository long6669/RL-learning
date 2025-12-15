# ====================================================
# First-Visit Monte Carlo Control with Exploring Starts
# ====================================================

import gymnasium as gym
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# 创建环境
env = gym.make('Blackjack-v1', render_mode=None)

# 超参数
EPISODES = 500000
GAMMA = 1.0
MIN_EPSILON = 0.01
DECAY_RATE = 0.9999

# 初始化数据结构
# Q：Q-Table
Q = defaultdict(lambda: np.zeros(env.action_space.n))

# N: 存储出现次数，计算平均值
N = defaultdict(lambda: np.zeros(env.action_space.n))

def get_action(state, epsilon):
    """Epsilon-Greedy 策略"""
    state_tuple = state
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state_tuple])

# ====================================================
# 训练循环
# ====================================================
print(f"🚀 开始训练：First-Visit Monte Carlo Control with {EPISODES} episodes...")

epsilon = 1.0

for episode in range(EPISODES):
    state, info = env.reset()
    state_tuple = state

    episode_history = []
    done = False

    while not done:
        action = get_action(state_tuple, epsilon)

        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        episode_history.append((state_tuple, action, reward))

        state_tuple = next_state
    # 回报G
    G = 0
    # 记录本局已经访问过的 (状态, 动作) 对
    visited_state_actions = set()

    for t, (s, a, r) in enumerate(reversed(episode_history)):
        G = r + GAMMA * G

        if (s, a) not in visited_state_actions:
            visited_state_actions.add((s, a))

            N[s][a] += 1

            N_s_a = N[s][a]
            Q[s][a] += (G - Q[s][a]) / N_s_a

    epsilon = max(MIN_EPSILON, epsilon * DECAY_RATE)

    if (episode + 1) % 50000 == 0:
        print(f"Episode: {episode + 1}, Epsilon: {epsilon:.4f}")

print("✅ 训练完成！Q-Table 已填充。")
env.close()

# 现在 Q 表里存的就是“最优策略”了。我们来跑 1000 局看看胜率
env_test = gym.make('Blackjack-v1', render_mode=None)
wins = 0
tests = 10000

for _ in range(tests):
    state, _ = env_test.reset()
    done = False
    
    while not done:
        # 完全贪婪策略 (epsilon=0)，只选择 Q 表中最好的动作
        action = np.argmax(Q[state])
        next_state, reward, terminated, truncated, _ = env_test.step(action)
        done = terminated or truncated
        state = next_state
        
    if reward == 1.0:
        wins += 1

env_test.close()
win_rate = wins / tests * 100

print(f"\n--- 评估结果 (MC Control) ---")
print(f"测试局数: {tests}")
# 专家级玩家的胜率通常在 42% 左右
print(f"AI 胜率: {win_rate:.2f}% (通常在 40% ~ 43% 视为成功)")


    


