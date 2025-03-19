import numpy as np
import random
import torch
import torch.nn as nn
import gc

class DeepQLearning:
    def __init__(self, env, gamma, epsilon, epsilon_min, epsilon_dec, episodes, batch_size, memory, model, max_steps, optimizer, loss_fn, target_model=None, update_target_every=10):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.episodes = episodes
        self.batch_size = batch_size
        self.memory = memory
        self.model = model
        self.max_steps = max_steps
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        # Use a separate target model if provided, otherwise use the online model.
        self.target_model = target_model if target_model is not None else model
        self.update_target_every = update_target_every

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.env.action_space.n)
        # Convert state to torch tensor (assuming state shape is (1, input_dim))
        state_tensor = torch.FloatTensor(state)
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
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        terminals = np.array([exp[4] for exp in batch]).astype(np.uint8)

        # Convert arrays to torch tensors
        states_tensor = torch.FloatTensor(states).squeeze(1)         # shape: (batch_size, input_dim)
        next_states_tensor = torch.FloatTensor(next_states).squeeze(1)   # shape: (batch_size, input_dim)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1)          # shape: (batch_size, 1)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
        terminals_tensor = torch.FloatTensor(terminals).unsqueeze(1)

        # Compute current Q-values for the batch of states using the online model
        self.model.train()
        q_values = self.model(states_tensor)

        # Double DQN modification:
        # 1. Use the online model to select the best next actions.
        # 2. Evaluate those actions using the target model.
        with torch.no_grad():
            next_actions = torch.argmax(self.model(next_states_tensor), dim=1, keepdim=True)
            next_q_values_target = self.target_model(next_states_tensor)
            next_q_values = torch.gather(next_q_values_target, 1, next_actions)

        # Compute the target Q-values
        targets = rewards_tensor + self.gamma * next_q_values * (1 - terminals_tensor)

        # Select the Q-values corresponding to the actions taken in the batch
        q_selected = torch.gather(q_values, 1, actions_tensor)

        # Compute loss and backpropagate
        loss = self.loss_fn(q_selected, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay the epsilon value to reduce exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_dec

    def train(self):
        rewards_all = []
        for i in range(self.episodes + 1):
            state, _ = self.env.reset()
            state = np.reshape(state, (1, self.env.observation_space.shape[0]))
            score = 0
            steps = 0
            done = False
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
                self.experience_replay()
                if done:
                    print(f'Episode: {i+1}/{self.episodes}. Score: {score}, Epsilon: {self.epsilon}')
                    break
            rewards_all.append(score)
            gc.collect()
            # Update the target network periodically
            if i % self.update_target_every == 0:
                self.target_model.load_state_dict(self.model.state_dict())
        return rewards_all
