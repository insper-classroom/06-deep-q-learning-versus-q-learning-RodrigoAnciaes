import gymnasium as gym
import numpy as np
import time
import argparse
import matplotlib.pyplot as plt
from matplotlib import animation

class MountainCarAgent:
    def __init__(self, env, num_bins=30):
        """Initialize the agent with the same discretization as the training agent"""
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.num_bins = num_bins
        self.bins = self._create_bins()
        self.q_table = None
    
    def _create_bins(self):
        """Create bins for discretizing the continuous state space"""
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
    
    def load_q_table(self, filename):
        """Load a saved Q-table from file"""
        try:
            self.q_table = np.load(filename)
            print(f"Successfully loaded Q-table from {filename}")
            print(f"Q-table shape: {self.q_table.shape}")
            return True
        except Exception as e:
            print(f"Error loading Q-table: {e}")
            return False
    
    def select_action(self, state):
        """Select the best action based on the current state"""
        if self.q_table is None:
            raise ValueError("Q-table not loaded. Call load_q_table first.")
        discrete_state = self._discretize_state(state)
        return np.argmax(self.q_table[discrete_state])

def test_agent(q_table_file, num_episodes=10, render=True, delay=0.01, save_video=False, video_filename=None):
    """Test a trained Q-learning agent"""
    render_mode = 'human' if render and not save_video else 'rgb_array' if save_video else None
    env = gym.make('MountainCar-v0', render_mode=render_mode)
    
    # Create agent and load Q-table
    agent = MountainCarAgent(env)
    if not agent.load_q_table(q_table_file):
        env.close()
        return
    
    rewards = []
    steps_list = []
    frames = []  # For video recording
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done:
            steps += 1
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Capture frame if saving video
            if save_video:
                frames.append(env.render())
            
            done = terminated or truncated or steps > 200
            state = next_state
            total_reward += reward
            
            if render and not save_video:
                time.sleep(delay)  # Slow down rendering for visualization
        
        rewards.append(total_reward)
        steps_list.append(steps)
        print(f"Episode {episode+1}: Reward = {total_reward:.2f}, Steps = {steps}")
    
    env.close()
    
    # Calculate statistics
    avg_reward = np.mean(rewards)
    avg_steps = np.mean(steps_list)
    print(f"\nTest Results:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Success Rate: {sum(r > -200 for r in rewards) / len(rewards) * 100:.2f}%")
    
    # Save video if requested
    if save_video and video_filename:
        save_frames_as_video(frames, video_filename)
    
    return rewards, steps_list

def display_q_table_heatmap(q_table_file, num_bins=30):
    """Display a heatmap of the Q-table values"""
    q_table = np.load(q_table_file)
    
    # Create a value map showing the maximum Q-value for each state
    value_map = np.max(q_table, axis=2)
    
    # Define the state space
    position_space = np.linspace(-1.2, 0.6, num_bins)
    velocity_space = np.linspace(-0.07, 0.07, num_bins)
    
    # Create a policy map showing the best action for each state
    # 0: push left, 1: no push, 2: push right
    policy_map = np.argmax(q_table, axis=2)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Value map (max Q-value)
    im1 = ax1.imshow(value_map, origin='lower', aspect='auto', cmap='viridis')
    ax1.set_title('Value Map (Max Q-value)')
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Velocity')
    ax1.set_xticks([0, num_bins//4, num_bins//2, 3*num_bins//4, num_bins-1])
    ax1.set_xticklabels([f'{position_space[0]:.1f}', f'{position_space[num_bins//4]:.1f}', 
                         f'{position_space[num_bins//2]:.1f}', f'{position_space[3*num_bins//4]:.1f}', 
                         f'{position_space[-1]:.1f}'])
    ax1.set_yticks([0, num_bins//4, num_bins//2, 3*num_bins//4, num_bins-1])
    ax1.set_yticklabels([f'{velocity_space[0]:.3f}', f'{velocity_space[num_bins//4]:.3f}', 
                         f'{velocity_space[num_bins//2]:.3f}', f'{velocity_space[3*num_bins//4]:.3f}', 
                         f'{velocity_space[-1]:.3f}'])
    plt.colorbar(im1, ax=ax1, label='Q-value')
    
    # Policy map (best action)
    cmap = plt.cm.get_cmap('coolwarm', 3)
    im2 = ax2.imshow(policy_map, origin='lower', aspect='auto', cmap=cmap, vmin=-0.5, vmax=2.5)
    ax2.set_title('Policy Map (Best Action)')
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Velocity')
    ax2.set_xticks([0, num_bins//4, num_bins//2, 3*num_bins//4, num_bins-1])
    ax2.set_xticklabels([f'{position_space[0]:.1f}', f'{position_space[num_bins//4]:.1f}', 
                         f'{position_space[num_bins//2]:.1f}', f'{position_space[3*num_bins//4]:.1f}', 
                         f'{position_space[-1]:.1f}'])
    ax2.set_yticks([0, num_bins//4, num_bins//2, 3*num_bins//4, num_bins-1])
    ax2.set_yticklabels([f'{velocity_space[0]:.3f}', f'{velocity_space[num_bins//4]:.3f}', 
                         f'{velocity_space[num_bins//2]:.3f}', f'{velocity_space[3*num_bins//4]:.3f}', 
                         f'{velocity_space[-1]:.3f}'])
    cbar = plt.colorbar(im2, ax=ax2, ticks=[0, 1, 2])
    cbar.set_label('Action')
    cbar.set_ticklabels(['Push Left', 'No Push', 'Push Right'])
    
    plt.tight_layout()
    plt.savefig('q_table_analysis.png', dpi=300)
    plt.show()

def save_frames_as_video(frames, filename='mountain_car.mp4', fps=30):
    """Save frames as a video file"""
    print(f"Saving video to {filename}...")
    
    # Set up the writer
    writer = animation.FFMpegWriter(fps=fps)
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Hide axes
    ax.set_axis_off()
    
    # Create the animation
    with writer.saving(fig, filename, dpi=100):
        for i, frame in enumerate(frames):
            ax.clear()
            ax.imshow(frame)
            ax.set_axis_off()
            writer.grab_frame()
            
            # Show progress
            if i % 20 == 0:
                print(f"Processing frame {i}/{len(frames)}")
    
    plt.close()
    print(f"Video saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test a trained Mountain Car Q-learning agent')
    parser.add_argument('--q_table', type=str, default='data/q_table_best.npy',
                        help='Path to the Q-table file')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to run')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment (display animation)')
    parser.add_argument('--delay', type=float, default=0.01,
                        help='Delay between frames when rendering (seconds)')
    parser.add_argument('--analyze', action='store_true',
                        help='Display heatmap analysis of the Q-table')
    parser.add_argument('--save_video', action='store_true',
                        help='Save a video of the agent performance')
    parser.add_argument('--video_filename', type=str, default='mountain_car_agent.mp4',
                        help='Filename for the saved video')
    
    args = parser.parse_args()
    
    # Test the agent
    if args.render or args.save_video:
        test_agent(args.q_table, args.episodes, args.render, args.delay, 
                   args.save_video, args.video_filename)
    
    # Analyze the Q-table
    if args.analyze:
        display_q_table_heatmap(args.q_table)
    
    # If no flags specified, show help
    if not (args.render or args.analyze or args.save_video):
        print("No action specified. Showing help message:")
        parser.print_help()
        print("\nExample usage:")
        print("  python test_mountain_car.py --render --episodes 3")
        print("  python test_mountain_car.py --analyze")
        print("  python test_mountain_car.py --save_video")