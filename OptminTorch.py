import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os
from tqdm import tqdm
import time

# Create directories if they don't exist
os.makedirs("results", exist_ok=True)
os.makedirs("data", exist_ok=True)

# OPTIMIZATION 1: Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

# OPTIMIZATION 2: Improved DeepQLearning class with batched processing
class OptimizedDeepQLearning:
    def __init__(self, env, gamma, epsilon, epsilon_min, epsilon_dec, episodes, batch_size, memory, model, max_steps, optimizer, loss_fn, target_model=None, update_target_every=10):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.episodes = episodes
        self.batch_size = batch_size
        self.memory = memory
        self.model = model.to(device)
        self.max_steps = max_steps
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        # Use a separate target model if provided, otherwise use the online model
        self.target_model = target_model.to(device) if target_model is not None else model
        self.update_target_every = update_target_every
        
        # OPTIMIZATION 3: Pre-allocate memory for tensors
        self.states_tensor = None
        self.next_states_tensor = None
        self.actions_tensor = None
        self.rewards_tensor = None
        self.terminals_tensor = None

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.env.action_space.n)
        
        # Convert state to torch tensor
        state_tensor = torch.FloatTensor(state).to(device)
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(state_tensor)
        action = torch.argmax(q_values, dim=1).item()
        return action

    def experience(self, state, action, reward, next_state, terminal):
        self.memory.append((state, action, reward, next_state, terminal))

    def experience_replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        # OPTIMIZATION 4: Sample once and reuse
        batch = random.sample(self.memory, self.batch_size)
        
        # Efficiently extract batch data
        states = np.vstack([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.vstack([exp[3] for exp in batch])
        terminals = np.array([exp[4] for exp in batch]).astype(np.float32)
        
        # OPTIMIZATION 5: Reuse tensors if possible to reduce memory allocations
        if self.states_tensor is None or self.states_tensor.shape[0] != states.shape[0]:
            # Create new tensors if needed
            self.states_tensor = torch.FloatTensor(states).to(device)
            self.next_states_tensor = torch.FloatTensor(next_states).to(device)
            self.actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(device)
            self.rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(device)
            self.terminals_tensor = torch.FloatTensor(terminals).unsqueeze(1).to(device)
        else:
            # Reuse existing tensors
            self.states_tensor.copy_(torch.FloatTensor(states))
            self.next_states_tensor.copy_(torch.FloatTensor(next_states))
            self.actions_tensor.copy_(torch.LongTensor(actions).unsqueeze(1))
            self.rewards_tensor.copy_(torch.FloatTensor(rewards).unsqueeze(1))
            self.terminals_tensor.copy_(torch.FloatTensor(terminals).unsqueeze(1))

        # Compute current Q-values
        self.model.train()
        q_values = self.model(self.states_tensor)
        
        # Double DQN implementation
        with torch.no_grad():
            next_actions = torch.argmax(self.model(self.next_states_tensor), dim=1, keepdim=True)
            next_q_values_target = self.target_model(self.next_states_tensor)
            next_q_values = torch.gather(next_q_values_target, 1, next_actions)
            
        # Compute target Q-values
        targets = self.rewards_tensor + self.gamma * next_q_values * (1 - self.terminals_tensor)
        
        # Select Q-values for actions taken
        q_selected = torch.gather(q_values, 1, self.actions_tensor)
        
        # Compute loss and backpropagate
        loss = self.loss_fn(q_selected, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    # OPTIMIZATION 6: Early stopping and more efficient training loop
    def train(self, early_stop_reward=-100, patience=50):
        rewards_all = []
        losses = []
        best_avg_reward = -float('inf')
        episodes_without_improvement = 0
        
        for i in tqdm(range(self.episodes)):
            state, _ = self.env.reset()
            state = np.reshape(state, (1, self.env.observation_space.shape[0]))
            score = 0
            steps = 0
            done = False
            episode_losses = []
            
            while not done:
                steps += 1
                action = self.select_action(state)
                next_state, reward, terminal, truncated, _ = self.env.step(action)
                if terminal or truncated or (steps > self.max_steps):
                    done = True
                    
                score += reward
                next_state = np.reshape(next_state, (1, self.env.observation_space.shape[0]))
                self.experience(state, action, reward, next_state, terminal)
                state = next_state
                
                if len(self.memory) >= self.batch_size:
                    loss = self.experience_replay()
                    if loss is not None:
                        episode_losses.append(loss)
                
                if done:
                    if self.epsilon > self.epsilon_min:
                        self.epsilon *= self.epsilon_dec
                    break
            
            # Record metrics
            rewards_all.append(score)
            if episode_losses:
                losses.append(np.mean(episode_losses))
            
            # Update target network periodically
            if i % self.update_target_every == 0:
                self.target_model.load_state_dict(self.model.state_dict())
            
            # Early stopping check
            if i >= 100:  # Start checking after 100 episodes
                current_avg_reward = np.mean(rewards_all[-100:])
                if current_avg_reward > best_avg_reward:
                    best_avg_reward = current_avg_reward
                    episodes_without_improvement = 0
                else:
                    episodes_without_improvement += 1
                
                # If reward threshold reached or no improvement for a while
                if (current_avg_reward >= early_stop_reward or 
                    episodes_without_improvement >= patience):
                    print(f"\nEarly stopping at episode {i+1}")
                    print(f"Average reward over last 100 episodes: {current_avg_reward:.2f}")
                    break
        
        return rewards_all, losses

def run_training(run_number, episodes=1000):
    # OPTIMIZATION 7: Create environment only once
    env = gym.make('MountainCar-v0')
    
    # Different seeds for each run
    seed = run_number * 100
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    print(f'\nRun {run_number+1}/5')
    print(f'State space: {env.observation_space}, Action space: {env.action_space}')

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    # Initialize model
    model = QNetwork(input_dim, output_dim)
    target_model = QNetwork(input_dim, output_dim)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    # OPTIMIZATION 8: Use better hyperparameters
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01  # Slightly higher than before
    epsilon_dec = 0.995  # Slower decay
    batch_size = 128  # Larger batch size
    memory = deque(maxlen=10000)
    max_steps = 200  # Reduced from 500
    
    # OPTIMIZATION 9: Use a higher learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    loss_fn = nn.MSELoss()

    # Create DQN agent
    start_time = time.time()
    DQN = OptimizedDeepQLearning(
        env, gamma, epsilon, epsilon_min, epsilon_dec, episodes, 
        batch_size, memory, model, max_steps, optimizer, loss_fn, 
        target_model=target_model, update_target_every=10
    )
    
    # Train with early stopping
    rewards, losses = DQN.train(early_stop_reward=-100, patience=1000)
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"average reward: {np.mean(rewards):.2f}")
    print(f"Training time for run {run_number+1}: {training_time:.2f} seconds")
    
    # Save model
    torch.save(model.state_dict(), f'data/model_Mountain_run_{run_number+1}.pth')
    
    # Save rewards
    with open(f'results/MountainCar_DeepQLearning_rewards_run_{run_number+1}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Reward'])
        for episode, reward in enumerate(rewards):
            writer.writerow([episode, reward])
    
    env.close()
    return rewards, training_time

if __name__ == "__main__":
    num_runs = 5
    episodes = 1000
    import random  # Make sure to import random
    
    # Store results from all runs
    all_rewards = []
    training_times = []
    
    overall_start = time.time()
    
    for run in range(num_runs):
        rewards, training_time = run_training(run, episodes)
        all_rewards.append(rewards)
        training_times.append(training_time)
    
    overall_end = time.time()
    overall_time = overall_end - overall_start
    
    # Convert to numpy array for easier manipulation
    all_rewards_np = [np.array(rewards) for rewards in all_rewards]
    
    # Find the shortest sequence length for proper alignment
    min_length = min(len(rewards) for rewards in all_rewards_np)
    all_rewards_aligned = [rewards[:min_length] for rewards in all_rewards_np]
    all_rewards_np = np.array(all_rewards_aligned)
    
    # Calculate average rewards across all runs
    avg_rewards = np.mean(all_rewards_np, axis=0)
    std_rewards = np.std(all_rewards_np, axis=0)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot individual runs
    for run in range(num_runs):
        plt.plot(all_rewards_aligned[run], alpha=0.3, 
                 label=f'Run {run+1} ({len(all_rewards[run])} episodes)')
    
    # Calculate rolling mean for smoothed plot
    window_size = min(20, min_length // 5)  # Adjust window size based on data
    avg_rewards_rolling = np.convolve(avg_rewards, np.ones(window_size)/window_size, mode='valid')
    episodes_rolling = np.arange(len(avg_rewards_rolling))
    
    # Plot average
    plt.plot(avg_rewards, color='blue', alpha=0.7, label='Average Reward')
    plt.plot(episodes_rolling, avg_rewards_rolling, color='red', linewidth=2, 
             label=f'Average Reward ({window_size}-episode rolling mean)')
    
    # Add shaded area for standard deviation
    plt.fill_between(np.arange(len(avg_rewards)), 
                     avg_rewards - std_rewards, 
                     avg_rewards + std_rewards,
                     alpha=0.2, color='blue')
    
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title(f'Mountain Car DQN: Average Performance Over {num_runs} Runs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add training time information
    avg_time = np.mean(training_times)
    time_text = f"Avg Training Time: {avg_time:.2f}s per run\nTotal Time: {overall_time:.2f}s"
    plt.figtext(0.02, 0.02, time_text, fontsize=10)
    
    # Save figure
    plt.savefig("results/MountainCar_DeepQLearning_optimized.jpg", dpi=300)
    plt.close()
    
    # Save average results
    with open('results/MountainCar_DeepQLearning_optimized_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Run', 'Episodes', 'Avg_Reward', 'Training_Time'])
        for run in range(num_runs):
            writer.writerow([run+1, len(all_rewards[run]), 
                            np.mean(all_rewards[run]), training_times[run]])
        # Add summary row
        writer.writerow(['Average', np.mean([len(r) for r in all_rewards]), 
                        np.mean([np.mean(r) for r in all_rewards]), avg_time])
    
    # Save the best model
    avg_rewards_per_run = [np.mean(rewards) for rewards in all_rewards]
    best_run = np.argmax(avg_rewards_per_run)
    print(f"\nBest run was run {best_run+1} with average reward {avg_rewards_per_run[best_run]:.2f}")
    
    # Copy the best model to a general filename
    import shutil
    shutil.copy(f'data/model_Mountain_run_{best_run+1}.pth', 'data/model_Mountain_best.pth')
    
    print("\nTraining complete!")
    print(f"Average training time: {avg_time:.2f} seconds per run")
    print(f"Total time for all runs: {overall_time:.2f} seconds")
    print(f"Results saved to results/MountainCar_DeepQLearning_optimized.jpg")