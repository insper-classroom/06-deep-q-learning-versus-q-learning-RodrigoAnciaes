import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from TorchDeepQ import DeepQLearning
import os

# Create directories if they don't exist
os.makedirs("results", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Define the Q-Network using PyTorch
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def run_training(run_number, episodes=1000):
    # Initialize environment and set seeds with different values for each run
    env = gym.make('MountainCar-v0')
    seed = run_number * 100  # Using different seeds for each run
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print(f'\nRun {run_number+1}/5:')
    print('State space: ', env.observation_space)
    print('Action space: ', env.action_space)

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    model = QNetwork(input_dim, output_dim)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    target_model = QNetwork(input_dim, output_dim)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    gamma = 0.99 
    epsilon = 1.0
    epsilon_min = 0.001
    epsilon_dec = 0.99
    batch_size = 64
    memory = deque(maxlen=10000)  # Experience replay memory
    max_steps = 500

    DQN = DeepQLearning(env, gamma, epsilon, epsilon_min, epsilon_dec, episodes, 
                        batch_size, memory, model, max_steps, optimizer, loss_fn, 
                        target_model=target_model, update_target_every=10)
    
    print(f"Starting training for run {run_number+1}")
    rewards = DQN.train()
    
    # Save model for each run
    torch.save(model.state_dict(), f'data/model_Mountain_run_{run_number+1}.pth')
    
    # Save rewards for each run
    with open(f'results/MountainCar_DeepQLearning_rewards_run_{run_number+1}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Reward'])  # Header
        for episode, reward in enumerate(rewards):
            writer.writerow([episode, reward])
    
    env.close()
    return rewards

if __name__ == "__main__":
    num_runs = 5
    episodes = 1000
    
    # Store rewards from all runs
    all_rewards = []
    
    for run in range(num_runs):
        rewards = run_training(run, episodes)
        all_rewards.append(rewards)
    
    # Convert to numpy array for easier manipulation
    all_rewards_np = np.array(all_rewards)
    
    # Calculate average rewards across all runs
    avg_rewards = np.mean(all_rewards_np, axis=0)
    
    # Calculate standard deviation for confidence intervals
    std_rewards = np.std(all_rewards_np, axis=0)
    
    # Plot average rewards with shaded area for standard deviation
    plt.figure(figsize=(12, 6))
    
    # Plot individual runs (thin lines)
    for run in range(num_runs):
        plt.plot(all_rewards[run], alpha=0.3, label=f'Run {run+1}' if run == 0 else None)
    
    # Calculate rolling mean for smoothed plot
    avg_rewards_rolling = np.convolve(avg_rewards, np.ones(20)/20, mode='valid')
    episodes_rolling = np.arange(len(avg_rewards_rolling))
    
    # Plot average with rolling mean
    plt.plot(avg_rewards, color='blue', alpha=0.5, label='Average Reward')
    plt.plot(episodes_rolling, avg_rewards_rolling, color='red', linewidth=2, label='Average Reward (20-episode rolling mean)')
    
    # Add shaded area for standard deviation
    plt.fill_between(np.arange(len(avg_rewards)), 
                    avg_rewards - std_rewards, 
                    avg_rewards + std_rewards,
                    alpha=0.2, color='blue')
    
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Mountain Car DQN: Average Rewards Over 5 Runs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.savefig("results/MountainCar_DeepQLearning_multiple_runs.jpg", dpi=300)
    plt.close()
    
    # Save combined average results
    with open('results/MountainCar_DeepQLearning_average_rewards.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Average_Reward', 'Std_Deviation'])  # Header
        for episode in range(len(avg_rewards)):
            writer.writerow([episode, avg_rewards[episode], std_rewards[episode]])
    
    # Save the best model (based on the run with highest average reward)
    avg_rewards_per_run = np.mean(all_rewards_np, axis=1)
    best_run = np.argmax(avg_rewards_per_run)
    print(f"\nBest run was run {best_run+1} with average reward {avg_rewards_per_run[best_run]:.2f}")
    
    # Copy the best model to a general filename
    import shutil
    shutil.copy(f'data/model_Mountain_run_{best_run+1}.pth', 'data/model_Mountain_best.pth')
    
    print("Training complete!")
    print(f"Average reward across all runs: {np.mean(avg_rewards):.2f}")
    print(f"Results saved to results/MountainCar_DeepQLearning_multiple_runs.jpg")