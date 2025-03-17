import numpy as np
import random
import torch
import torch.nn as nn
import gc

class DeepQLearning:
    def __init__(self, env, gamma, epsilon, epsilon_min, epsilon_dec, episodes, batch_size, memory, model, max_steps, optimizer, loss_fn):
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

        # Convert to torch tensors and squeeze the extra dimension
        states_tensor = torch.FloatTensor(states).squeeze(1)         # Now shape: (batch_size, input_dim)
        next_states_tensor = torch.FloatTensor(next_states).squeeze(1)  # Now shape: (batch_size, input_dim)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1)         # shape: (batch_size, 1)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
        terminals_tensor = torch.FloatTensor(terminals).unsqueeze(1)

        # Compute current Q values for the batch of states
        self.model.train()
        q_values = self.model(states_tensor)  # Expected shape: (batch_size, num_actions)

        # Compute Q values for next states (without gradient tracking)
        with torch.no_grad():
            next_q_values = self.model(next_states_tensor)
        next_max, _ = torch.max(next_q_values, dim=1, keepdim=True)

        # Compute the target Q values
        targets = rewards_tensor + self.gamma * next_max * (1 - terminals_tensor)

        # Select Q values corresponding to the actions taken
        q_selected = torch.gather(q_values, 1, actions_tensor)

        loss = self.loss_fn(q_selected, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
                    print(f'Episode: {i+1}/{self.episodes}. Score: {score}')
                    break
            rewards_all.append(score)
            gc.collect()
        return rewards_all
