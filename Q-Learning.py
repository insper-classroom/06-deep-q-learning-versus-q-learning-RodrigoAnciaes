import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import csv
import time
import os
import random
from tqdm import tqdm

# Create directories if they don't exist
os.makedirs("results", exist_ok=True)
os.makedirs("data", exist_ok=True)

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
        
        for i in tqdm(range(self.episodes)):
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
                
                # Custom reward shaping to speed up learning (optional)
                # Reward for getting closer to the goal
                position, velocity = next_state
                # Reward for moving right with positive velocity
                shaped_reward = reward
                shaped_reward += 10 * abs(velocity)  # Additional reward for speed
                
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
                
                # If reward threshold reached or no improvement for a while
                if (current_avg_reward >= early_stop_reward or 
                    episodes_without_improvement >= patience):
                    print(f"\nEarly stopping at episode {i+1}")
                    print(f"Average reward over last 100 episodes: {current_avg_reward:.2f}")
                    break
        
        return rewards_all
    
    def save_q_table(self, filename):
        """Save Q-table to file"""
        np.save(filename, self.q_table)
    
    def load_q_table(self, filename):
        """Load Q-table from file"""
        self.q_table = np.load(filename)

def run_training(run_number, episodes=1000):
    """Run a single training session"""
    # Create environment
    env = gym.make('MountainCar-v0')
    
    # Set seeds for reproducibility
    seed = run_number * 100
    np.random.seed(seed)
    random.seed(seed)
    env.action_space.seed(seed)
    
    print(f'\nRun {run_number+1}/5')
    print(f'State space: {env.observation_space}, Action space: {env.action_space}')
    
    # Q-learning hyperparameters
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
        episodes, max_steps, num_bins
    )
    
    rewards = agent.train(early_stop_reward=-110, patience=10000)
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training time for run {run_number+1}: {training_time:.2f} seconds")
    
    # Save Q-table
    agent.save_q_table(f'data/q_table_run_{run_number+1}.npy')
    
    # Save rewards
    with open(f'results/MountainCar_QLearning_rewards_run_{run_number+1}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Reward'])
        for episode, reward in enumerate(rewards):
            writer.writerow([episode, reward])
    
    env.close()
    return rewards, training_time

if __name__ == "__main__":
    num_runs = 5
    episodes = 10000
    
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
    
    # Find the shortest sequence length for proper alignment
    min_length = min(len(rewards) for rewards in all_rewards)
    all_rewards_aligned = [rewards[:min_length] for rewards in all_rewards]
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
    plt.title(f'Mountain Car Q-Learning: Average Performance Over {num_runs} Runs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add training time information
    avg_time = np.mean(training_times)
    time_text = f"Avg Training Time: {avg_time:.2f}s per run\nTotal Time: {overall_time:.2f}s"
    plt.figtext(0.02, 0.02, time_text, fontsize=10)
    
    # Save figure
    plt.savefig("results/MountainCar_QLearning_multiple_runs.jpg", dpi=300)
    plt.close()
    
    # Save average results
    with open('results/MountainCar_QLearning_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Run', 'Episodes', 'Avg_Reward', 'Training_Time'])
        for run in range(num_runs):
            writer.writerow([run+1, len(all_rewards[run]),
                            np.mean(all_rewards[run]), training_times[run]])
        # Add summary row
        writer.writerow(['Average', np.mean([len(r) for r in all_rewards]),
                        np.mean([np.mean(r) for r in all_rewards]), avg_time])
    
    # Identify the best run
    avg_rewards_per_run = [np.mean(rewards) for rewards in all_rewards]
    best_run = np.argmax(avg_rewards_per_run)
    print(f"\nBest run was run {best_run+1} with average reward {avg_rewards_per_run[best_run]:.2f}")
    
    # Copy the best Q-table to a general filename
    import shutil
    shutil.copy(f'data/q_table_run_{best_run+1}.npy', 'data/q_table_best.npy')
    
    print("\nTraining complete!")
    print(f"Average training time: {avg_time:.2f} seconds per run")
    print(f"Total time for all runs: {overall_time:.2f} seconds")
    print(f"Results saved to results/MountainCar_QLearning_multiple_runs.jpg")