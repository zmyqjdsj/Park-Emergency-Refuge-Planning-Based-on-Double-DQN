import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

G={1: [[6, 40.0], [7, 40.0], [8, 40.0], [9, 40.0], [10, 40.0]],
 2: [[4, 43.5], [5, 43.5], [6, 43.5], [7, 43.5]],
 3: [[9, 70.0], [10, 42.5]],
 4: [[5, 39.0], [7, 87.0]],
 5: [[6, 87.0]],
 6: [[7, 43.0], [10, 80.0], [12, 58.0], [13, 40.0]],
 7: [[8, 30.0]],
 8: [[9, 81.0]],
 9: [[10, 70.0]],
 10: [[11, 115.0], [13, 40.0]],
 11: [[13, 35.0]],
 12: [[13, 10.0]],
 13: []}

source = 3#起点
sink = 13#终点
n_nodes = len(G)
# DQN参数
gamma = 0.999
learning_rate = 0.001
memory_size = 1000
batch_size = 128
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.75


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def get_possible_actions(state):
    return G[state]


def get_next_state(state, action):
    return action[0]


def get_reward(state, action):
    return -action[1]


def get_state_vector(state):
    state_vector = torch.zeros(n_nodes)
    state_vector[state - 1] = 1
    return state_vector.unsqueeze(0)


def choose_action(state, epsilon):
    possible_actions = get_possible_actions(state)

    if (np.random.rand() <= epsilon):

        return random.choice(possible_actions)
    else:
        state_vector = get_state_vector(state)
        q_values = model(state_vector).detach()
        action_q_values = [q_values[0, a[0] - 1] for a in possible_actions]
        best_action_idx = np.argmax(action_q_values)
        return possible_actions[best_action_idx]


def replay(buffer, optimizer, criterion):
    if len(buffer) < batch_size:
        return

    batch = random.sample(buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.cat(states)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float).unsqueeze(1)
    next_states = torch.cat(next_states)
    dones = torch.tensor(dones, dtype=torch.float).unsqueeze(1)

    optimizer.zero_grad()

    q_values = model(states)
    next_q_values = model(next_states).detach()

    max_next_q_values, _ = torch.max(next_q_values, 1)
    max_next_q_values = max_next_q_values.unsqueeze(1)
    target_q_values = q_values.clone()
    target_q_values.scatter_(1, torch.tensor([[a[0] - 1] for a in actions]),
                             rewards + gamma * max_next_q_values * (1 - dones))

    loss = criterion(q_values, target_q_values)
    loss.backward()
    optimizer.step()


# 初始化DQN模型、优化器和损失函数
model = DQN(n_nodes, n_nodes)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# 训练参数
n_episodes = 5000
max_steps_per_episode = 100

# 初始化记忆缓冲区
memory = deque(maxlen=memory_size)

# 开始训练
for episode in range(n_episodes):
    state = source
    total_reward = 0

    for step in range(max_steps_per_episode):
        action = choose_action(state, epsilon)
        next_state = get_next_state(state, action)
        reward = get_reward(state, action)
        done = next_state == sink

        memory.append((get_state_vector(state), action, reward, get_state_vector(next_state), done))

        total_reward += reward
        state = next_state

        if done:
            break

    replay(memory, optimizer, criterion)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if episode % 100 == 0:
        print(f'Episode {episode}, Total Reward: {total_reward}')

# 训练完成后，展示最短路径
state = source
shortest_path = [state]
while state != sink:
    action = choose_action(state, 0)  # 设置epsilon为0，以便始终选择最佳动作
    state = get_next_state(state, action)
    shortest_path.append(state)

print(f'Shortest path: {shortest_path}')
