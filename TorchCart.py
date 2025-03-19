import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from TorchDeepQ import DeepQLearning

# Define the Q-Network using PyTorch
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

env = gym.make('CartPole-v1')
np.random.seed(0)
torch.manual_seed(0)

print('State space: ', env.observation_space)
print('Action space: ', env.action_space)

input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

model = QNetwork(input_dim, output_dim)
print(model)

optimizer = optim.Adam(model.parameters(), lr=0.00007)
loss_fn = nn.MSELoss()

target_model = QNetwork(input_dim, output_dim)
target_model.load_state_dict(model.state_dict())
target_model.eval()


gamma = 0.99 
epsilon = 1.0
epsilon_min = 0.001
epsilon_dec = 0.999
episodes = 1000
batch_size = 64
memory = deque(maxlen=10000)  # Experience replay memory
max_steps = 500

DQN = DeepQLearning(env, gamma, epsilon, epsilon_min, epsilon_dec, episodes, batch_size, memory, model, max_steps, optimizer, loss_fn, target_model=target_model, update_target_every=10)
rewards = DQN.train()

plt.plot(rewards)
plt.xlabel('Episodes')
plt.ylabel('# Rewards')
plt.title('# Rewards vs Episodes')
plt.savefig("results/cartpole_DeepQLearning.jpg")
plt.close()

with open('results/cartpole_DeepQLearning_rewards.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    for episode, reward in enumerate(rewards):
        writer.writerow([episode, reward])

torch.save(model.state_dict(), 'data/model_cart_pole.pth')
