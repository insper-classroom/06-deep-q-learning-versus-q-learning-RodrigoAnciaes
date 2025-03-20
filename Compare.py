import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from tqdm import tqdm
import argparse
import csv

# Create directories if they don't exist
os.makedirs("results", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("comparison", exist_ok=True)

# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#############################
# Q-Learning Implementation #
#############################
class QLearning:
    def __init__(self, env, learning_rate, gamma, epsilon, epsilon_min, epsilon_dec, episodes, max_steps, num_bins=20):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.episodes = episodes
        self.max_steps = max_steps
        
        # Get state and action spaces
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        
        # Create bins for discretizing continuous state space
        self.num_bins = num_bins
        self.bins = self._create_bins()
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((self.num_bins, self.num_bins, self.action_size))
    
    def _create_bins(self):
        bins = []
        for i in range(self.state_size):
            bins.append(np.linspace(
                self.env.observation_space.low[i],
                self.env.observation_space.high[i],
                self.num_bins + 1
            ))
        return bins
    
    def _discretize_state(self, state):
        """Convert continuous state to discrete state indices"""
        indices = []
        for i, s in enumerate(state):
            # Find the bin index for each dimension
            index = np.digitize(s, self.bins[i]) - 1
            # Ensure the index is within bounds
            index = min(self.num_bins - 1, max(0, index))
            indices.append(index)
        return tuple(indices)
    
    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        
        # Get discretized state and return best action
        discrete_state = self._discretize_state(state)
        return np.argmax(self.q_table[discrete_state])
    
    def update_q_table(self, state, action, reward, next_state, done):
        """Update Q-table using the Q-learning update rule"""
        discrete_state = self._discretize_state(state)
        discrete_next_state = self._discretize_state(next_state)
        
        # Q-learning formula
        current_q = self.q_table[discrete_state + (action,)]
        
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[discrete_next_state])
            target_q = reward + self.gamma * max_next_q
        
        # Update Q-value
        self.q_table[discrete_state + (action,)] += self.learning_rate * (target_q - current_q)
    
    def train(self, early_stop_reward=-110, patience=50):
        """Train the Q-learning agent"""
        rewards_all = []
        best_avg_reward = -float('inf')
        episodes_without_improvement = 0
        
        for i in tqdm(range(self.episodes), desc="Q-Learning"):
            state, _ = self.env.reset()
            score = 0
            steps = 0
            done = False
            
            while not done:
                steps += 1
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                
                # Check if episode is done
                if terminated or truncated or (steps > self.max_steps):
                    done = True
                
                # Custom reward shaping to speed up learning
                position, velocity = next_state
                shaped_reward = reward
                if position > state[0]:  # Moving right
                    shaped_reward += 0.1
                if velocity > 0:  # Moving with positive velocity
                    shaped_reward += 0.1 * velocity
                
                # Update Q-table
                self.update_q_table(state, action, shaped_reward, next_state, done)
                
                # Update state and accumulate reward
                state = next_state
                score += reward  # Use original reward for scoring
                
                if done:
                    # Decay epsilon
                    if self.epsilon > self.epsilon_min:
                        self.epsilon *= self.epsilon_dec
                    break
            
            # Record reward
            rewards_all.append(score)
            
            # Early stopping check
            if i >= 100:  # Start checking after 100 episodes
                current_avg_reward = np.mean(rewards_all[-100:])
                if current_avg_reward > best_avg_reward:
                    best_avg_reward = current_avg_reward
                    episodes_without_improvement = 0
                else:
                    episodes_without_improvement += 1
                
                if (current_avg_reward >= early_stop_reward or 
                    episodes_without_improvement >= patience):
                    print(f"\nQ-Learning early stopping at episode {i+1}")
                    print(f"Average reward over last 100 episodes: {current_avg_reward:.2f}")
                    break
        
        return rewards_all

###########################
# DQN Implementation      #
###########################

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

class DeepQLearning:
    def __init__(self, env, gamma, epsilon, epsilon_min, epsilon_dec, episodes, batch_size, memory_size, max_steps, update_target_every=10):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.episodes = episodes
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.max_steps = max_steps
        self.update_target_every = update_target_every
        
        # Initialize models
        self.input_dim = env.observation_space.shape[0]
        self.output_dim = env.action_space.n
        self.model = QNetwork(self.input_dim, self.output_dim).to(device)
        self.target_model = QNetwork(self.input_dim, self.output_dim).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.002)
        self.loss_fn = nn.MSELoss()
        
        # Pre-allocate tensors
        self.states_tensor = None
        self.next_states_tensor = None
        self.actions_tensor = None
        self.rewards_tensor = None
        self.terminals_tensor = None

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.output_dim)
        
        # Convert state to torch tensor
        state_tensor = torch.FloatTensor(state).to(device)
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(state_tensor)
        action = torch.argmax(q_values, dim=1).item()
        return action

    def experience_replay(self):
        if len(self.memory) < self.batch_size:
            return None
            
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        
        # Extract batch data
        states = np.vstack([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.vstack([exp[3] for exp in batch])
        terminals = np.array([exp[4] for exp in batch]).astype(np.float32)
        
        # Create or reuse tensors
        if self.states_tensor is None or self.states_tensor.shape[0] != states.shape[0]:
            self.states_tensor = torch.FloatTensor(states).to(device)
            self.next_states_tensor = torch.FloatTensor(next_states).to(device)
            self.actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(device)
            self.rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(device)
            self.terminals_tensor = torch.FloatTensor(terminals).unsqueeze(1).to(device)
        else:
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

    def train(self, early_stop_reward=-100, patience=50):
        rewards_all = []
        losses = []
        best_avg_reward = -float('inf')
        episodes_without_improvement = 0
        
        for i in tqdm(range(self.episodes), desc="DQN"):
            state, _ = self.env.reset()
            state = np.reshape(state, (1, self.input_dim))
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
                    
                # Add to memory
                next_state = np.reshape(next_state, (1, self.input_dim))
                self.memory.append((state, action, reward, next_state, done))
                
                # Update state and score
                state = next_state
                score += reward
                
                # Experience replay
                if len(self.memory) >= self.batch_size:
                    loss = self.experience_replay()
                    if loss is not None:
                        episode_losses.append(loss)
                
                if done:
                    # Decay epsilon
                    if self.epsilon > self.epsilon_min:
                        self.epsilon *= self.epsilon_dec
                    break
            
            # Record reward for this episode
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
                
                if (current_avg_reward >= early_stop_reward or 
                    episodes_without_improvement >= patience):
                    print(f"\nDQN early stopping at episode {i+1}")
                    print(f"Average reward over last 100 episodes: {current_avg_reward:.2f}")
                    break
        
        return rewards_all, losses

def run_qlearning(run_id, seed=None, max_episodes=1000):
    """Run a single Q-learning experiment"""
    env = gym.make('MountainCar-v0')
    
    # Set seed for reproducibility if provided
    if seed is not None:
        seed_value = seed + run_id * 100
        np.random.seed(seed_value)
        random.seed(seed_value)
        env.action_space.seed(seed_value)
    
    # Hyperparameters
    learning_rate = 0.1
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_dec = 0.995
    max_steps = 200
    num_bins = 30  # Higher resolution for better performance
    
    # Create and train agent
    start_time = time.time()
    agent = QLearning(
        env, learning_rate, gamma, epsilon, epsilon_min, epsilon_dec,
        max_episodes, max_steps, num_bins
    )
    
    # Train agent
    rewards = agent.train(early_stop_reward=-110, patience=10000)
    
    # Calculate training time
    training_time = time.time() - start_time
    
    # Save Q-table and rewards
    agent.q_table.dump(f'data/comparison_q_table_run_{run_id}.npy')
    
    with open(f'comparison/q_learning_rewards_run_{run_id}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Reward'])
        for episode, reward in enumerate(rewards):
            writer.writerow([episode, reward])
    
    env.close()
    return rewards, training_time

def run_dqn(run_id, seed=None, max_episodes=1000):
    """Run a single DQN experiment"""
    env = gym.make('MountainCar-v0')
    
    # Set seed for reproducibility if provided
    if seed is not None:
        seed_value = seed + run_id * 100
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        random.seed(seed_value)
        env.action_space.seed(seed_value)
    
    # Hyperparameters
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_dec = 0.995
    batch_size = 128
    memory_size = 10000
    max_steps = 200
    update_target_every = 10
    
    # Create and train agent
    start_time = time.time()
    agent = DeepQLearning(
        env, gamma, epsilon, epsilon_min, epsilon_dec, max_episodes,
        batch_size, memory_size, max_steps, update_target_every
    )
    
    # Train agent
    rewards, losses = agent.train(early_stop_reward=-100, patience=10000)
    
    # Calculate training time
    training_time = time.time() - start_time
    
    # Save model and rewards
    torch.save(agent.model.state_dict(), f'data/comparison_dqn_model_run_{run_id}.pth')
    
    with open(f'comparison/dqn_rewards_run_{run_id}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Reward'])
        for episode, reward in enumerate(rewards):
            writer.writerow([episode, reward])
    
    env.close()
    return rewards, training_time

def load_rewards_from_csv(filename):
    """Load rewards from a CSV file"""
    rewards = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            rewards.append(float(row[1]))
    return rewards

def plot_comparison(q_learning_rewards, dqn_rewards, q_times, dqn_times, smoothing_window=20):
    """Plot comparison of learning curves"""
    plt.figure(figsize=(15, 10))
    
    # Create two subplots
    plt.subplot(2, 1, 1)
    
    # Plot individual Q-learning runs
    for i, rewards in enumerate(q_learning_rewards):
        plt.plot(rewards, alpha=0.3, color='blue')
    
    # Plot individual DQN runs
    for i, rewards in enumerate(dqn_rewards):
        plt.plot(rewards, alpha=0.3, color='red')
    
    # Find lengths for proper alignment
    q_min_length = min(len(rewards) for rewards in q_learning_rewards)
    dqn_min_length = min(len(rewards) for rewards in dqn_rewards)
    
    # Calculate mean rewards for both methods
    q_rewards_aligned = [rewards[:q_min_length] for rewards in q_learning_rewards]
    dqn_rewards_aligned = [rewards[:dqn_min_length] for rewards in dqn_rewards]
    
    q_mean_rewards = np.mean(np.array(q_rewards_aligned), axis=0)
    dqn_mean_rewards = np.mean(np.array(dqn_rewards_aligned), axis=0)
    
    # Calculate smoothed rewards
    q_window = min(smoothing_window, q_min_length // 5)
    dqn_window = min(smoothing_window, dqn_min_length // 5)
    
    q_smoothed = np.convolve(q_mean_rewards, np.ones(q_window)/q_window, mode='valid')
    dqn_smoothed = np.convolve(dqn_mean_rewards, np.ones(dqn_window)/dqn_window, mode='valid')
    
    # Plot mean rewards
    plt.plot(q_smoothed, linewidth=2, color='blue', label=f'Q-Learning ({len(q_learning_rewards)} runs)')
    plt.plot(dqn_smoothed, linewidth=2, color='red', label=f'DQN ({len(dqn_rewards)} runs)')
    
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Learning Curves: Q-Learning vs DQN for Mountain Car')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add a table with statistics in the second subplot
    plt.subplot(2, 1, 2)
    plt.axis('off')
    
    # Calculate statistics
    q_final_rewards = [np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards) for rewards in q_learning_rewards]
    dqn_final_rewards = [np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards) for rewards in dqn_rewards]
    
    q_avg_time = np.mean(q_times)
    dqn_avg_time = np.mean(dqn_times)
    
    q_avg_episodes = np.mean([len(rewards) for rewards in q_learning_rewards])
    dqn_avg_episodes = np.mean([len(rewards) for rewards in dqn_rewards])
    
    # Create a table with statistics
    data = [
        ['Metric', 'Q-Learning', 'DQN'],
        ['Avg. Final Reward', f'{np.mean(q_final_rewards):.2f}', f'{np.mean(dqn_final_rewards):.2f}'],
        ['Std. Final Reward', f'{np.std(q_final_rewards):.2f}', f'{np.std(dqn_final_rewards):.2f}'],
        ['Avg. Training Time (s)', f'{q_avg_time:.2f}', f'{dqn_avg_time:.2f}'],
        ['Avg. Episodes to Convergence', f'{q_avg_episodes:.1f}', f'{dqn_avg_episodes:.1f}'],
    ]
    
    table = plt.table(cellText=data, loc='center', cellLoc='center', colWidths=[0.3, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Save figure
    plt.tight_layout()
    plt.savefig('comparison/mountain_car_q_learning_vs_dqn.png', dpi=300)
    plt.savefig('comparison/mountain_car_q_learning_vs_dqn.pdf')
    plt.close()
    
    # Also create a unified plot with episodes normalized
    plt.figure(figsize=(10, 6))
    
    # Normalize episode lengths
    max_episodes = max(q_min_length, dqn_min_length)
    x_q = np.linspace(0, 1, len(q_mean_rewards))
    x_dqn = np.linspace(0, 1, len(dqn_mean_rewards))
    
    # Plot normalized curves
    plt.plot(x_q, q_mean_rewards, linewidth=2, color='blue', alpha=0.7, label='Q-Learning')
    plt.plot(x_dqn, dqn_mean_rewards, linewidth=2, color='red', alpha=0.7, label='DQN')
    
    plt.xlabel('Normalized Training Progress')
    plt.ylabel('Reward')
    plt.title('Q-Learning vs DQN: Normalized Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add a text box with key statistics
    stats = f"Q-Learning: Avg. Final Reward = {np.mean(q_final_rewards):.2f}, Time = {q_avg_time:.2f}s\n"
    stats += f"DQN: Avg. Final Reward = {np.mean(dqn_final_rewards):.2f}, Time = {dqn_avg_time:.2f}s"
    plt.figtext(0.5, 0.01, stats, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('comparison/mountain_car_normalized_comparison.png', dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Compare Q-Learning and DQN for Mountain Car')
    parser.add_argument('--runs', type=int, default=5, help='Number of runs for each algorithm (default: 5)')
    parser.add_argument('--episodes', type=int, default=1000, help='Maximum number of episodes per run (default: 1000)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--load', action='store_true', help='Load previously saved results instead of running new experiments')
    
    args = parser.parse_args()
    
    if args.load:
        # Load previously saved results
        print("Loading previously saved results...")
        
        q_learning_rewards = []
        dqn_rewards = []
        q_times = []
        dqn_times = []
        
        # Try to load the timing information
        try:
            with open('comparison/timing_info.csv', 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if row[0].startswith('q_learning'):
                        q_times.append(float(row[1]))
                    elif row[0].startswith('dqn'):
                        dqn_times.append(float(row[1]))
        except FileNotFoundError:
            print("Warning: Could not find timing information. Using placeholder values.")
            q_times = [100.0] * args.runs
            dqn_times = [150.0] * args.runs
        
        # Load the reward data
        for i in range(args.runs):
            try:
                q_rewards = load_rewards_from_csv(f'comparison/q_learning_rewards_run_{i}.csv')
                q_learning_rewards.append(q_rewards)
            except FileNotFoundError:
                print(f"Warning: Could not find Q-learning rewards for run {i}")
            
            try:
                dqn_rewards = load_rewards_from_csv(f'comparison/dqn_rewards_run_{i}.csv')
                dqn_rewards.append(dqn_rewards)
            except FileNotFoundError:
                print(f"Warning: Could not find DQN rewards for run {i}")
    else:
        # Run new experiments
        print(f"Running {args.runs} experiments for each algorithm, with up to {args.episodes} episodes each.")
        
        q_learning_rewards = []
        dqn_rewards = []
        q_times = []
        dqn_times = []
        
        # Run Q-learning experiments
        for i in range(args.runs):
            print(f"\nRunning Q-learning experiment {i+1}/{args.runs}")
            rewards, training_time = run_qlearning(i, seed=args.seed, max_episodes=args.episodes)
            q_learning_rewards.append(rewards)
            q_times.append(training_time)
            print(f"Training time: {training_time:.2f}s, Episodes: {len(rewards)}, Avg reward: {np.mean(rewards):.2f}")
        
        # Run DQN experiments
        for i in range(args.runs):
            print(f"\nRunning DQN experiment {i+1}/{args.runs}")
            rewards, training_time = run_dqn(i, seed=args.seed, max_episodes=args.episodes)
            dqn_rewards.append(rewards)
            dqn_times.append(training_time)
            print(f"Training time: {training_time:.2f}s, Episodes: {len(rewards)}, Avg reward: {np.mean(rewards):.2f}")
        
        # Save timing information
        with open('comparison/timing_info.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Algorithm', 'Training_Time'])
            for i, time_val in enumerate(q_times):
                writer.writerow([f'q_learning_{i}', time_val])
            for i, time_val in enumerate(dqn_times):
                writer.writerow([f'dqn_{i}', time_val])
    
    # Plot comparison
    print("\nPlotting comparison...")
    plot_comparison(q_learning_rewards, dqn_rewards, q_times, dqn_times)
    
    print("\nComparison complete! Results saved to the 'comparison' directory.")

if __name__ == "__main__":
    main()