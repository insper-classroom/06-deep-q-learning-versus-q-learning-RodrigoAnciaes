import gymnasium as gym
import numpy as np
import torch
import time
from TorchDeepQ import DeepQLearning
from TorchMontain import QNetwork

# Load the environment
env = gym.make('MountainCar-v0', render_mode='human')

# Set random seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Initialize model with the same architecture used during training
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
model = QNetwork(input_dim, output_dim)

# Load the trained model weights
model.load_state_dict(torch.load('data/model_Mountain_run_1.pth'))
model.eval()  # Set the model to evaluation mode

# Function to select actions using the trained model
def select_action(state, model):
    state_tensor = torch.FloatTensor(state)
    with torch.no_grad():
        q_values = model(state_tensor)
    return torch.argmax(q_values, dim=1).item()

# Test the model
def test_agent(num_episodes=5, max_steps=1000, delay=0.01):
    total_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = np.reshape(state, (1, env.observation_space.shape[0]))
        
        episode_reward = 0
        done = False
        steps = 0
        
        print(f"Episode {episode+1}/{num_episodes}")
        
        while not done and steps < max_steps:
            # Render the environment
            env.render()
            
            # Select action
            action = select_action(state, model)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Reshape next state
            next_state = np.reshape(next_state, (1, env.observation_space.shape[0]))
            
            # Update state and accumulate reward
            state = next_state
            episode_reward += reward
            steps += 1
            
            # Small delay to make the visualization viewable
            #time.sleep(delay)
            
            # Print progress
            if steps % 50 == 0:
                print(f"  Step {steps}, Position: {state[0][0]:.4f}, Velocity: {state[0][1]:.4f}")
        
        # Print episode summary
        print(f"  Episode {episode+1} finished after {steps} steps with reward {episode_reward}")
        print(f"  Final position: {state[0][0]:.4f}, Final velocity: {state[0][1]:.4f}")
        print()
        
        total_rewards.append(episode_reward)
    
    # Print overall performance
    print(f"Average reward over {num_episodes} episodes: {np.mean(total_rewards):.2f}")
    return total_rewards

if __name__ == "__main__":
    print("Testing the trained DQN agent on Mountain Car environment")
    print("Press Ctrl+C to stop the simulation")
    try:
        rewards = test_agent(num_episodes=5, delay=0.01)
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    finally:
        env.close()