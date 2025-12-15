
from termios import TAB0
from PIL.GimpGradientFile import EPSILON
import gymnasium as gym
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# 创建环境
env = gym.make('Blackjack-v1', render_mode=None)

# 超参数
EPISODES = 100000
GAMMA = 1.0
MIN_EPSILON = 0.01
DECAY_RATE = 0.9999
LEARNING_RATE = 0.01  # alpha: 学习率 (每次更新步子迈多大)

# 初始化数据结构
# Q：Q-Table
Q = defaultdict(lambda: np.zeros(env.action_space.n))

def get_action(state, epsilon):
    state_tuple = state
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state_tuple])
    
# ====================================================
# 训练循环
# ====================================================
print(f"🚀 开始训练：Q-learning with {EPISODES} episodes...")

epsilon = 1.0

for episode in range(EPISODES):
    state, info = env.reset()
    
    done = False

    while not done:
        action = get_action(state, epsilon)

        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        current_q = Q[state][action]

        if done:
            Target = reward
        else:
            Target = reward + GAMMA * np.max(Q[next_state])

        Q[state][action] += LEARNING_RATE * (Target - current_q)

        state = next_state

    epsilon = max(MIN_EPSILON, epsilon * DECAY_RATE)

    if (episode + 1) % 50000 == 0:
        print(f"Episode: {episode + 1}, Epsilon: {epsilon:.4f}")

print("✅ 训练完成！")
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

print(f"\n--- 评估结果 (Q-learning) ---")
print(f"测试局数: {tests}")
# 专家级玩家的胜率通常在 42% 左右
print(f"AI 胜率: {win_rate:.2f}% (通常在 40% ~ 43% 视为成功)")



